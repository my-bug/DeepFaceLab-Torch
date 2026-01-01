import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
import models
from core.interact import interact as io

def trainerThread (s2c, c2s, e,
                    model_class_name = None,
                    saved_models_path = None,
                    training_data_src_path = None,
                    training_data_dst_path = None,
                    pretraining_data_path = None,
                    pretrained_model_path = None,
                    no_preview=False,
                    force_model_name=None,
                    force_gpu_idxs=None,
                    cpu_only=None,
                    silent_start=False,
                    execute_programs = None,
                    max_iters: int = 0,
                    debug=False,
                    **kwargs):
    while True:
        try:
            start_time = time.time()

            save_interval_min = 25

            if not training_data_src_path.exists():
                training_data_src_path.mkdir(exist_ok=True, parents=True)

            if not training_data_dst_path.exists():
                training_data_dst_path.mkdir(exist_ok=True, parents=True)

            if not saved_models_path.exists():
                saved_models_path.mkdir(exist_ok=True, parents=True)
                            
            model = models.import_model(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        pretrained_model_path=pretrained_model_path,
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        force_gpu_idxs=force_gpu_idxs,
                        cpu_only=cpu_only,
                        silent_start=silent_start,
                        debug=debug)

            start_iter = model.get_iter()
            if max_iters is None:
                max_iters = 0
            try:
                max_iters = int(max_iters)
            except Exception:
                max_iters = 0

            is_reached_goal = model.is_reached_iter_goal()

            shared_state = { 'after_save' : False }
            loss_string = ""
            save_iter =  model.get_iter()
            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info ("正在保存……", end='\r')
                    model.save()
                    shared_state['after_save'] = True
                    
            def model_backup():
                if not debug and not is_reached_goal:
                    model.create_backup()             

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
                e.set() #Set the GUI Thread as Ready

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('模型已训练到目标迭代次数，可以使用预览。')
                else:
                    io.log_info('开始训练。目标迭代次数：%d。按“回车（Enter）”停止训练并保存模型。' % ( model.get_target_iter()  ) )
            else:
                io.log_info('开始训练。按“回车（Enter）”停止训练并保存模型。')

            last_save_time = time.time()

            # execute_programs is optional; normalize to a safe iterable.
            if execute_programs is None:
                execute_programs = []
            elif not isinstance(execute_programs, (list, tuple)):
                execute_programs = []

            _normalized_execute_programs = []
            for x in execute_programs:
                try:
                    prog_time, prog = x[0], x[1]
                except Exception:
                    continue
                _normalized_execute_programs.append([prog_time, prog, time.time()])
            execute_programs = _normalized_execute_programs

            for i in itertools.count(0,1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time)  >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("无法执行程序片段: %s" % (prog) )

                    if not is_reached_goal:

                        if model.get_iter() == 0:
                            io.log_info("")
                            io.log_info("正在尝试进行首次迭代。如果出现错误，请降低模型参数/配置。")
                            io.log_info("")
                            
                            if sys.platform[0:3] == 'win':
                                io.log_info("!!!")
                                io.log_info("Windows 10 用户重要提示：为保证正常运行，请按下图进行设置。")
                                io.log_info("https://i.imgur.com/B7cmDCB.jpg")
                                io.log_info("!!!")

                        iter, iter_time = model.train_one_iter()

                        if (not is_reached_goal) and (not debug) and max_iters > 0 and (iter - start_iter) >= max_iters:
                            io.log_info(f'\n已达到 max_iters={max_iters}。正在保存并停止。')
                            model_save()
                            is_reached_goal = True
                            send_preview()
                            i = -1
                            break

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]".format ( time_str, iter, '{:0.4f}'.format(iter_time) )
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]".format ( time_str, iter, int(iter_time*1000) )

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            
                            mean_loss = np.mean ( loss_history[save_iter:iter], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info (loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info ('\r' + loss_string, end='')
                            else:
                                io.log_info (loss_string, end='\r')

                        if model.get_iter() == 1:
                            model_save()

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('已达到目标迭代次数。')
                            model_save()
                            is_reached_goal = True
                            io.log_info ('现在可以使用预览。')
                
                need_save = False
                while time.time() - last_save_time >= save_interval_min*60:
                    last_save_time += save_interval_min*60
                    need_save = True
                
                if not is_reached_goal and need_save:
                    model_save()
                    send_preview()

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break



            model.finalize()

        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )



def main(**kwargs):
    io.log_info ("正在运行训练器。\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs )
    thread.start()

    e.wait() #Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        # Some OpenCV builds / Windows desktop setups may intermittently report a newly created
        # window as not visible, which would instantly trigger a clean shutdown.
        # Provide a debounce and an escape hatch via env vars.
        disable_preview_autoclose = str(os.environ.get('DFL_DISABLE_TRAIN_PREVIEW_AUTOCLOSE', '')).strip().lower() in (
            '1', 'y', 'yes', 'true', 'on'
        )
        try:
            preview_autoclose_grace_sec = float(os.environ.get('DFL_TRAIN_PREVIEW_AUTOCLOSE_GRACE_SEC', '2.0'))
        except Exception:
            preview_autoclose_grace_sec = 2.0
        preview_loop_start_time = time.time()
        invisible_consecutive = 0

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        close_requested = False
        close_requested_time = 0.0
        while True:
            # If user closed the window via OS controls, request a clean shutdown.
            if (not close_requested) and (not disable_preview_autoclose) and is_showing:
                try:
                    # OpenCV returns < 1 when window is closed/hidden.
                    if (time.time() - preview_loop_start_time) >= preview_autoclose_grace_sec:
                        if cv2.getWindowProperty(wnd_name, cv2.WND_PROP_VISIBLE) < 1:
                            invisible_consecutive += 1
                        else:
                            invisible_consecutive = 0

                        # Debounce: require a few consecutive reads to avoid false positives.
                        if invisible_consecutive >= 3:
                            io.log_info(
                                '检测到训练预览窗口已关闭/不可见，正在保存并退出… '
                                '(可设置环境变量 DFL_DISABLE_TRAIN_PREVIEW_AUTOCLOSE=1 禁用该自动退出)'
                            )
                            close_requested = True
                            close_requested_time = time.time()
                            s2c.put({'op': 'close'})
                except Exception:
                    pass

            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h,w,c) = selected_preview_rgb.shape

                # HEAD
                if close_requested:
                    elapsed = max(0.0, time.time() - close_requested_time)
                    head_lines = [
                        f'正在保存并退出…请稍候（{elapsed:0.1f}s）',
                        '[Ctrl+C]：强制退出',
                        ' ',
                    ]
                else:
                    head_lines = [
                        '[s]：保存  [b]：备份  [回车]：退出',
                        '[p]：刷新  [空格]：下一个预览  [l]：切换历史范围',
                        '预览："%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                        ]
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1

                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height
                    head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )

                final = head

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate ( [final, lh_img], axis=0 )

                final = np.concatenate ( [final, selected_preview_rgb], axis=0 )
                final = np.clip(final, 0, 1)

                io.show_image( wnd_name, (final*255).astype(np.uint8) )
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if (key == ord('\n') or key == ord('\r') or key == 27) and not close_requested:
                # 回车 / Esc -> 请求保存并退出
                io.log_info('收到退出按键（Enter/Esc），正在保存并退出…')
                close_requested = True
                close_requested_time = time.time()
                update_preview = True
                s2c.put ( {'op': 'close'} )
            elif (not close_requested) and key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif (not close_requested) and key == ord('b'):
                s2c.put ( {'op': 'backup'} )
            elif (not close_requested) and key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif (not close_requested) and key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif (not close_requested) and key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()