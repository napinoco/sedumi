% generate_stepdif_oracle.m
%
% Runs the real Octave stepdif.m across several hand-built scenarios
% chosen to exercise its many branches (tpmtd sign, R.sd sign, c(1)
% sign, |t|>|tg|, and the y0+t*dy0<=0 break point).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_stepdif_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'stepdif');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 41);

function save_case(out_dir, name, d, R, y0, x, y, z, dy0, dx, dy, dz, b, mint, tpmtd)
    [t, rcdx] = stepdif(d, R, y0, x, y, z, dy0, dx, dy, dz, b, mint, tpmtd);
    save('-v7', fullfile(out_dir, [name '.mat']), ...
         'd', 'R', 'y0', 'x', 'y', 'z', 'dy0', 'dx', 'dy', 'dz', 'b', 'mint', 'tpmtd', ...
         't', 'rcdx');
    fprintf('  %s: t=%g rcdx=%g\n', name, t, rcdx);
end

m = 4; nvar = 6;

d.l = [1.3; 0.8; 0.5];
R.sd = 0.15;
R.b0 = 1.2;
R.w = [0.4; 0.6];
y0 = 2.0;
x = rand(nvar,1) + 0.5;
y = rand(m,1) - 0.5;
z = rand(nvar,1) + 0.5;
dy0 = -0.4;
dx = 0.1*(rand(nvar,1)-0.5);
dy = 0.1*(rand(m,1)-0.5);
dz = 0.1*(rand(nvar,1)-0.5);
b = rand(m,1) - 0.5;
mint = -0.5;

% Case 1: tpmtd > 0, R.sd > 0 (usegap branch)
save_case(out_dir, 'case1_tpmtd_pos_sdpos', d, R, y0, x, y, z, dy0, dx, dy, dz, b, mint, 1);

% Case 2: tpmtd < 0, R.sd > 0
save_case(out_dir, 'case2_tpmtd_neg_sdpos', d, R, y0, x, y, z, dy0, dx, dy, dz, b, mint, -1);

% Case 3: R.sd < 0, dRg computed sign varies -> usegap likely false
R3 = R; R3.sd = -0.15;
save_case(out_dir, 'case3_sdneg', d, R3, y0, x, y, z, dy0, dx, dy, dz, b, mint, 1);

% Case 4: R.sd exactly 0
R4 = R; R4.sd = 0;
save_case(out_dir, 'case4_sdzero', d, R4, y0, x, y, z, dy0, dx, dy, dz, b, mint, 1);

% Case 5: larger dx/dy0 magnitude to push |t|>|tg| and the y0+t*dy0<=0
% break point.
dy0_5 = -1.8;
dx5 = (rand(nvar,1)-0.5);
dz5 = (rand(nvar,1)-0.5);
dy5 = (rand(m,1)-0.5);
save_case(out_dir, 'case5_large_step', d, R, y0, x, y, z, dy0_5, dx5, dy5, dz5, b, mint, 1);

% Case 6: negative tpmtd with the large-step scenario too.
save_case(out_dir, 'case6_large_step_neg_tpmtd', d, R, y0, x, y, z, dy0_5, dx5, dy5, dz5, b, mint, -1);

% Case 7: pushes into the interior-minimizer branch (c(1)>0, t<0 side)
% via a large dx(1) relative to x(1).
xA = [2;1;1;1;1;1]; zA = [2;1;1;1;1;1];
yA = [0.1;-0.2;0.3;-0.1]; bA = [0.1;-0.2;0.3;-0.1];
dxA = [3;0;0;0;0;0]; dzA = [0.01;0;0;0;0;0]; dyA = [0;0;0;0]; dy0A = 0.5;
save_case(out_dir, 'case7_interior_minimizer', d, R, y0, xA, yA, zA, dy0A, dxA, dyA, dzA, bA, mint, 1);

% Case 8: large negative dy0 triggers the y0+t*dy0<=0 break-point clamp
% (t = -y0/dy0).
dy0B = -10;
save_case(out_dir, 'case8_breakpoint', d, R, y0, xA, yA, zA, dy0B, dxA, dyA, dzA, bA, mint, 1);

fprintf('stepdif oracle written to %s\n', out_dir);
