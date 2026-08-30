% generate_wregion_oracle.m
%
% Runs the real Octave wregion.m on the same mixed L+Lorentz+real-SDP
% problem/ADA factorization as the recent pcg/sddir fixtures, covering:
%   - wr.delta == 0 (no initial centering; the actual first-iteration
%     state, since sedumi.m itself inits wr.delta=0.0)
%   - wr.delta > 0 (initial centering path)
%   - pars.stepdif == 1 (exercises the stepdif+trydif 2nd correction)
%     and pars.stepdif ~= 1 (skips it)
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_wregion_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

global ADA_sedumi_

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
dense.q = zeros(0,1);
dense.l = 0;

[d, v, vfrm, y, y0, R] = sdinit(At2, b2, c2, dense, K2, pars);
d.q1 = d.q1 .* (1 + 0.1*rand(size(d.q1)));
d.q2 = d.q2 + 0.05*(rand(size(d.q2))-0.5);

Ablkjc = partitA(At2, K2.mainblks);
DAt = getDAtm(At2, Ablkjc, dense, sparse(m,0), d, K2);

ADA = zeros(m,m);
for j = 1:m
    ej = zeros(m,1); ej(j) = 1;
    Aej = Amul(At2, dense, ej, 1);
    PAej = PopK(d, Aej, K2);
    ADA(:,j) = Amul(At2, dense, PAej, 0);
end
ADA = sparse((ADA+ADA')/2);
ADA_sedumi_ = ADA;

L = symbchol();
[L.L, L.d, L.skip, L.add] = blkchol(L, ADA);
L.d(find(L.skip)) = inf;

Lden.betajc = 1;

Lsd = sdfactor(L, Lden, dense, DAt, d, v, y, At2, c2, K2, R, y0, pars);

function save_case(out_dir, name, L, Lden, Lsd, d, v, vfrm, At2, DAt, dense, R, K2, y, y0, b2, pars, wr)
    [xscl,yout,zscl,y0out, w,relt, dxmdz,err, wrout] = ...
        wregion(L,Lden,Lsd,d,v,vfrm,At2,DAt,dense, R,K2,y,y0,b2, pars, wr);
    save('-v7', fullfile(out_dir, [name '.mat']), ...
         'K2', 'd', 'v', 'vfrm', 'At2', 'DAt', 'dense', 'R', 'y', 'y0', 'b2', 'pars', 'wr', 'Lsd', ...
         'xscl', 'yout', 'zscl', 'y0out', 'w', 'relt', 'dxmdz', 'err', 'wrout');
    fprintf('  %s: y0out=%g relt.p=%g relt.d=%g\n', name, y0out, relt.p, relt.d);
end

Lstruct.L = L.L; Lstruct.d = L.d; Lstruct.xsuper = L.xsuper; Lstruct.perm = L.perm;

% Case 1: wr.delta == 0 (real first-iteration state), pars.stepdif ~= 1.
wr1.delta = 0.0; wr1.desc = 1;
pars1 = pars; pars1.stepdif = 2;
save_case(out_dir, 'wregion_case1', L, Lden, Lsd, d, v, vfrm, At2, DAt, dense, R, K2, y, y0, b2, pars1, wr1);

% Case 2: wr.delta > 0 (initial centering path), pars.stepdif == 1
% (exercises stepdif+trydif).
wr2.delta = 0.2; wr2.h = 1.5; wr2.alpha = 0.05; wr2.desc = 1;
pars2 = pars; pars2.stepdif = 1;
save_case(out_dir, 'wregion_case2', L, Lden, Lsd, d, v, vfrm, At2, DAt, dense, R, K2, y, y0, b2, pars2, wr2);

save('-v7', '-append', fullfile(out_dir, 'wregion_case1.mat'), 'Lstruct');
save('-v7', '-append', fullfile(out_dir, 'wregion_case2.mat'), 'Lstruct');

fprintf('wregion oracle written to %s\n', out_dir);
