%% Regenerate Live Scripts (.mlx) from .m files
% Run this script inside MATLAB when source .m files are updated.

clear; clc;

files = {'problem1', 'problem2', 'problem3'};

for k = 1:numel(files)
    src = [files{k} '.m'];
    dst = [files{k} '.mlx'];
    if isfile(src)
        matlab.internal.liveeditor.openAndSave(src, dst);
        fprintf('Generated %s\n', dst);
    else
        warning('Missing source file: %s', src);
    end
end

fprintf('Done.\n');

