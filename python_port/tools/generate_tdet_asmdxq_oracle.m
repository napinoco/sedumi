% generate_tdet_asmdxq_oracle.m
%
% Runs the real Octave tdet.m and asmDxq.m using a genuine scaling-point
% struct `d` from sdinit.m (reusing the same mixed L+Lorentz+real-SDP
% problem as generate_sdinit_oracle.m), plus a perturbed x so the
% Lorentz-block quantities are not all-identical.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_tdet_asmdxq_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 77);

K.f = 0; K.l = 2; K.q = [3;4]; K.r = zeros(1,0); K.s = [2;3];
n = K.l + sum(K.q) + K.s(1)^2 + K.s(2)^2;
m = 4;
At = rand(n,m) - 0.5;
b = rand(m,1) - 0.5;
c = rand(n,1) - 0.5;
pars = checkpars(struct());

[At2, b2, c2, K2, prep] = pretransfo(At, b, c, K, pars);

dense.cols = zeros(0,1);
dense.A = sparse(length(b2), 0);

[d, v, vfrm, y, y0, R] = sdinit(At2, b2, c2, dense, K2, pars);

% Perturb x away from the trivial identity-multiple v, but keep it a
% generic full-length internal-format vector (same shape as v).
x = v + 0.05 * (rand(size(v)) - 0.5);

tdetx = tdet(x, K2);

[y_asm2, t_asm2] = asmDxq(d, x, K2);      % nargout==2 branch
y_asm1 = asmDxq(d, x, K2);                % nargout==1 branch (extra d.q1/d.q2 term added)

% Also exercise the "ddotx given directly" (nargin==4) branch, and the
% "only q-part given" (length(x) < K.lq) branch.
ddotx_precomputed = d.q1 .* x(K2.mainblks(1):K2.mainblks(2)-1) + ...
    ddot(d.q2, x, K2.qblkstart);
y_asm_ddotx = asmDxq(d, x, K2, ddotx_precomputed);

qonly_len = length(K2.q) + (K2.qblkstart(end) - K2.qblkstart(1));
x_qonly = x(K2.mainblks(1):K2.mainblks(1)+qonly_len-1);
[y_asm_qonly, t_asm_qonly] = asmDxq(d, x_qonly, K2);

save('-v7', fullfile(out_dir, 'tdet_asmdxq.mat'), ...
     'K2', 'd', 'x', 'tdetx', 'y_asm2', 't_asm2', 'y_asm1', ...
     'ddotx_precomputed', 'y_asm_ddotx', 'x_qonly', 'y_asm_qonly', 't_asm_qonly');

fprintf('tdet/asmDxq oracle written to %s\n', out_dir);
