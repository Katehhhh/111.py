'''
===========================================
  @author:  jayce
  @file:    extract_video_keyframes_av.py
  @time:    2022/4/11   21:42
===========================================
'''

import av
import os
import shutil

path_to_video = r'C:\Users\yyy\Desktop\bird11.mp4'
output_dir = r'D:\pyav视频关键帧提取'


# # 提取全部帧
# container = av.open(path_to_video)
#
# for frame in container.decode(video=0):
#     frame.to_image().save(r'E:\Code\Python\比例尺鉴定\20220410比例尺鉴定\extract_video_keyframes\pyav\frame-%04d.png' % frame.index)


def extract_video_keyframes(path_to_video, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 提取关键帧
        with av.open(path_to_video) as container:
            # 表示我们只想查看关键帧
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = 'NONKEY'
            #设置了一个a计数，用来只保存到第二张关键帧
            a=1
            for frame in container.decode(stream):
                print(frame)
                # 使用frame.pts的原因是frame.index对skip_frame没有意义,因为关键帧是从所有的帧中抽取中独立的图像，而pts显示的就是这些独立图像的index；
                # DTS（Decoding Time Stamp）：即解码时间戳，这个时间戳的意义在于告诉播放器该在什么时候解码这一帧的数据。
                # PTS（Presentation Time Stamp）：即显示时间戳，这个时间戳用来告诉播放器该在什么时候显示这一帧的数据。
                frame.to_image().save(os.path.join(output_dir, 'temporary-image-1.jpg'.format(frame.pts)))
                #最终结果为保存的是第二张关键帧
                if(a == 0):
                    break
                a -= 1

    except Exception as e:
        print('Program error occurred:{}'.format(repr(e)))


if __name__ == "__main__":
    extract_video_keyframes(path_to_video, output_dir)
    # shutil.rmtree(output_dir)
