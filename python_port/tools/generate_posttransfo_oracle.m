% generate_posttransfo_oracle.m
%
% Runs the real Octave posttransfo.m, reusing the already-committed
% pretransfo fixtures (for their K2/prep.QR) so this doesn't need a
% fresh pretransfo call -- covers a real-only case (lorentz.mat, no
% K.ycomplex) and the K.ycomplex case (ycomplex.mat) directly.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_posttransfo_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
pretransfo_dir = fullfile(pkg_root, 'tests', 'fixtures', 'pretransfo');
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'posttransfo');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 66);

function save_case(out_dir, name, K2, prep, x, y)
    [xp, yp, Kp] = posttransfo(x, y, prep, K2);
    save('-v7', fullfile(out_dir, [name '.mat']), ...
         'K2', 'prep', 'x', 'y', 'xp', 'yp', 'Kp');
    fprintf('  %s: length(xp)=%d length(yp)=%d Kp.l=%d\n', name, length(xp), length(yp), Kp.l);
end

% Case 1: real-only (lorentz.mat has no K.ycomplex)
data1 = load(fullfile(pretransfo_dir, 'lorentz.mat'), 'K2', 'prep');
x1 = rand(data1.K2.N, 1) - 0.5;
y1 = rand(data1.K2.m, 1) - 0.5;
save_case(out_dir, 'real_only', data1.K2, data1.prep, x1, y1);

% Case 2: K.ycomplex nonempty
data2 = load(fullfile(pretransfo_dir, 'ycomplex.mat'), 'K2', 'prep');
x2 = rand(data2.K2.N, 1) - 0.5;
y2 = rand(data2.K2.m, 1) - 0.5;
save_case(out_dir, 'with_ycomplex', data2.K2, data2.prep, x2, y2);

% Case 3: complex SDP block (complex_sdp_single.mat) -- exercises a
% genuinely complex prep.QR, so x comes back complex.
data3 = load(fullfile(pretransfo_dir, 'complex_sdp_single.mat'), 'K2', 'prep');
x3 = rand(data3.K2.N, 1) - 0.5;
y3 = rand(data3.K2.m, 1) - 0.5;
save_case(out_dir, 'complex_sdp', data3.K2, data3.prep, x3, y3);

fprintf('posttransfo oracle written to %s\n', out_dir);
