% generate_pcg_oracle.m
%
% Runs the real Octave wrapPcg.m and loopPcg.m on a genuine
% A*P(d)*A' system, built for the same mixed L+Lorentz+real-SDP problem
% used by the recent sdinit/getDAtm/PopK fixtures. ADA = A*P(d)*A' is
% built via dense column-by-column evaluation (Amul+PopK on unit
% vectors) rather than getada1/getada2/getada3 (still deferred -- part
% of the Phase 2 "dense columns"/ADA-construction subsystem), which
% gives byte-for-byte the same matrix any correct construction method
% must produce, since it's simply the definition of the linear operator
% A*P(d)*A' evaluated exactly.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_pcg_oracle"

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

% Build ADA = A*P(d)*A' column by column.
ADA = zeros(m,m);
for j = 1:m
    ej = zeros(m,1); ej(j) = 1;
    Aej = Amul(At2, dense, ej, 1);
    PAej = PopK(d, Aej, K2);
    ADA(:,j) = Amul(At2, dense, PAej, 0);
end
ADA = sparse((ADA+ADA')/2);   % symmetrize away roundoff asymmetry
ADA_sedumi_ = ADA;

L = symbchol();
[L.L, L.d, L.skip, L.add] = blkchol(L, ADA);
L.d(find(L.skip)) = inf;

Lden.betajc = 1;   % no dense columns

pars.cg = checkpars(struct()).cg;

% ---- wrapPcg ----
rb = rand(m,1) - 0.5;
rv = rand(K2.N,1) - 0.5;   % rv is a FULL cone-variable-space vector (length N), not the spectral dim
[wy, wdx, wk, wr] = wrapPcg(L, Lden, At2, dense, d, DAt, K2, rb, rv, pars.cg, min(1,y0)*R.maxRb);

% ---- loopPcg (fresh start, p=[]) ----
bvec = rand(m,1) - 0.5;
restol = 1e-10;
[ly, lk, lDAy] = loopPcg(L, Lden, At2, dense, d, DAt, K2, bvec, [], 0, pars.cg, restol);

save('-v7', fullfile(out_dir, 'pcg.mat'), ...
     'At2', 'K2', 'd', 'dense', 'DAt', 'ADA', ...
     'rb', 'rv', 'y0', 'R', 'wy', 'wdx', 'wk', 'wr', ...
     'bvec', 'restol', 'ly', 'lk', 'lDAy');

Lstruct.L = L.L; Lstruct.d = L.d; Lstruct.xsuper = L.xsuper; Lstruct.perm = L.perm;
save('-v7', '-append', fullfile(out_dir, 'pcg.mat'), 'Lstruct');

fprintf('PCG oracle written to %s\n', out_dir);
