import os, sys, datetime
p = r'D:\大三下\语音pro\out_full_vc.wav'
print('path:', p)
print('exists:', os.path.exists(p))
if os.path.exists(p):
    st = os.stat(p)
    print('size_bytes:', st.st_size)
    print('mtime:', datetime.datetime.fromtimestamp(st.st_mtime).isoformat())
    print('ctime:', datetime.datetime.fromtimestamp(st.st_ctime).isoformat())
else:
    print('file not found')
