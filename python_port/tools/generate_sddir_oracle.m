% generate_sddir_oracle.m
%
% Runs the real Octave sdfactor.m and sddir.m (pMode 1, 2, 3) on the
% same mixed L+Lorentz+real-SDP problem/scaling point/ADA factorization
% as generate_pcg_oracle.m.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_sddir_oracle"

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

% ---- sdfactor ----
Lsd = sdfactor(L, Lden, dense, DAt, d, v, y, At2, c2, K2, R, y0, pars);

% ---- sddir, pMode 1, 2, 3 ----
n_spectral = length(vfrm.lab);
pv1 = rand(n_spectral,1) - 0.5;
[dx1,dy1,dz1,dy01,err1] = sddir(L,Lden,Lsd,pv1, d,v,vfrm,At2,DAt,dense, R,K2,y,y0,b2, pars,1);

[dx2,dy2,dz2,dy02,err2] = sddir(L,Lden,Lsd,[], d,v,vfrm,At2,DAt,dense, R,K2,y,y0,b2, pars,2);

pv3 = rand(K2.N,1) - 0.5;
[dx3,dy3,dz3,dy03,err3] = sddir(L,Lden,Lsd,pv3, d,v,vfrm,At2,DAt,dense, R,K2,y,y0,b2, pars,3);

save('-v7', fullfile(out_dir, 'sddir.mat'), ...
     'At2', 'b2', 'c2', 'K2', 'd', 'dense', 'DAt', 'v', 'y', 'y0', 'R', 'vfrm', 'pars', ...
     'Lsd', 'pv1', 'dx1', 'dy1', 'dz1', 'dy01', 'err1', ...
     'dx2', 'dy2', 'dz2', 'dy02', 'err2', ...
     'pv3', 'dx3', 'dy3', 'dz3', 'dy03', 'err3');

Lstruct.L = L.L; Lstruct.d = L.d; Lstruct.xsuper = L.xsuper; Lstruct.perm = L.perm;
save('-v7', '-append', fullfile(out_dir, 'sddir.mat'), 'Lstruct');

fprintf('sddir/sdfactor oracle written to %s\n', out_dir);
