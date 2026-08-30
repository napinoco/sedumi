% generate_dpr1fact_oracle.m
%
% Generates the dpr1fact/fwdpr1/bwdpr1 oracle fixtures
% (tests/fixtures/cluster2/dpr1fact_n{1,2}.mat) using the real Octave/MEX
% build.
%
% dpr1fact.c's "Lsym.dz" input is not a plain sparse matrix in the usual
% sense: it is a "cumulative compact row set" scheme where each column's
% stored nonzeros are only the rows *newly introduced* relative to
% previous columns (so dz.jc[n] <= m always, never sum of per-column
% nnz). This was reverse-engineered from dpr1fact.c/fwdpr1.c directly,
% not from any doc comment. The n=1 case sidesteps this entirely (a
% single column's own nonzero rows are automatically a valid "cumulative"
% set); the n=2 case builds it by hand: column 1 introduces 2 new rows,
% column 2 introduces 2 more (not overlapping column 1's), matching what
% the real finsymbden()/incorder() pipeline would produce for a nested
% elimination front.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_dpr1fact_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cluster2');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 7);

%% n=1: single dense column.
m = 5; n = 1;
LAD = sparse(rand(m,n));
perm = (1:n)';
dz = spones(LAD);
firstq = n+1;
Lsym = finsymbden(LAD, perm, dz, firstq);
d = rand(m,1) + 2;
smult = rand(n,1);
maxu = 10;
[Lden, Ld] = dpr1fact(LAD, d, Lsym, smult, maxu);
Lden.dz = Lsym.dz; Lden.first = Lsym.first; Lden.perm = Lsym.perm;
b = rand(m,1);
y = fwdpr1(Lden, b);
z = bwdpr1(Lden, b);
save('-v7', fullfile(out_dir, 'dpr1fact_n1.mat'), ...
     'LAD', 'd', 'smult', 'maxu', 'dz', 'perm', 'firstq', 'Lden', 'Ld', 'b', 'y', 'z');
fprintf('wrote dpr1fact_n1.mat\n');

%% n=2: two dense columns, dz's compact rows deliberately nested/disjoint
%% (see the module docstring above) rather than each column's own
%% independent sparsity pattern.
rand('seed', 3);
m = 5; n = 2;
LAD = sparse(m,n);
LAD([1,2],1) = rand(2,1);
LAD([1,2,3,4],2) = rand(4,1);
perm = (1:n)';
dz = sparse(m,n);
dz([1,2],1) = 1;    % column 1: newly-introduced rows {1,2}
dz([3,4],2) = 1;    % column 2: newly-introduced rows {3,4} (NOT repeating 1,2)
firstq = n+1;
Lsym = finsymbden(LAD, perm, dz, firstq);
d = rand(m,1) + 2;
smult = rand(n,1);
maxu = 10;
[Lden, Ld] = dpr1fact(LAD, d, Lsym, smult, maxu);
Lden.dz = Lsym.dz; Lden.first = Lsym.first; Lden.perm = Lsym.perm;
b = rand(m,1);
y = fwdpr1(Lden, b);
z = bwdpr1(Lden, b);
save('-v7', fullfile(out_dir, 'dpr1fact_n2.mat'), ...
     'LAD', 'd', 'smult', 'maxu', 'dz', 'perm', 'firstq', 'Lden', 'Ld', 'b', 'y', 'z');
fprintf('wrote dpr1fact_n2.mat\n');
