import os
paths = []
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'voice_env', 'Lib', 'site-packages'))
for root, dirs, files in os.walk(root_dir):
    if 'platforms' in dirs:
        paths.append(os.path.join(root, 'platforms'))
if not paths:
    print('NO_PLATFORMS_FOUND')
else:
    for p in paths:
        print(p)
