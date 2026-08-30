% generate_optstep_oracle.m
%
% Runs the real Octave optstep.m on a pure-LP problem (K.f=0, K.q=[],
% K.s=[] -- the only scope this port's optstep.py implements, since
% it's the only scope sedumi.m's own `lponly` gate ever reaches).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_optstep_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

global ADA_sedumi_

rand('seed', 101);

K.f = 0; K.l = 6; K.q = zeros(1,0); K.r = zeros(1,0); K.s = zeros(1,0);
n = K.l;
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

Ablkjc = partitA(At2, K2.mainblks);
DAt = getDAtm(At2, Ablkjc, dense, sparse(length(b2),0), d, K2);

ADA = zeros(length(b2), length(b2));
for j = 1:length(b2)
    ej = zeros(length(b2),1); ej(j) = 1;
    Aej = Amul(At2, dense, ej, 1);
    PAej = PopK(d, Aej, K2);
    ADA(:,j) = Amul(At2, dense, PAej, 0);
end
ADA = sparse((ADA+ADA')/2);
ADA_sedumi_ = ADA;

L = symbchol();
Ltmpsiz = L.tmpsiz;
[L.L, L.d, L.skip, L.add] = blkchol(L, ADA);
L.tmpsiz = Ltmpsiz;

symLden = symbcholden(L, dense, DAt);

% Generic direction with mixed signs (any vector works for lpNB=find(dxmdz<0);
% optstep.m doesn't require dxmdz to be a genuine PCG output for this test).
dxmdz = rand(K2.N,1) - 0.5;

feasratio = 0.95;

[xsol, ysol] = optstep(At2, b2, c2, y0, y, d, v, dxmdz, K2, L, symLden, dense, ...
    Ablkjc, struct(), ADA_sedumi_, DAt, feasratio, R, pars);

save('-v7', fullfile(out_dir, 'optstep.mat'), ...
     'K2', 'At2', 'b2', 'c2', 'y0', 'y', 'd', 'v', 'dxmdz', 'L', 'dense', 'R', 'pars', ...
     'feasratio', 'xsol', 'ysol');

if isempty(xsol)
    fprintf('optstep oracle written to %s (xsol empty -- rejected)\n', out_dir);
else
    fprintf('optstep oracle written to %s (xsol found, x0=%g)\n', out_dir, xsol(1));
end
