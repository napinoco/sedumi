% generate_blkchol_oracle.m
%
% Runs the REAL Octave/MEX symbchol()+blkchol() (the actual numeric LDL'
% factorization) on each fixture matrix in tests/fixtures/blkchol/*.mat,
% and saves L.L (numeric values), L.d, skip, diagadd next to it. This is
% the oracle python_port/tests/test_blkchol.py checks the ctypes
% symbolic_cholesky()+numeric_cholesky() bindings against.
%
% symbchol() reads its input from the global ADA_sedumi_ (SeDuMi's own
% convention, to avoid a MATLAB/Octave copy of a possibly-huge sparse
% matrix) -- this script follows that same convention rather than
% reimplementing symbchol.m's plumbing.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_blkchol_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
fixture_dir = fullfile(pkg_root, 'tests', 'fixtures', 'blkchol');

addpath(repo_root);

global ADA_sedumi_

files = dir(fullfile(fixture_dir, '*_spd.mat'));
for i = 1:numel(files)
  name = files(i).name;
  path = fullfile(fixture_dir, name);
  data = load(path, 'X');
  ADA_sedumi_ = data.X;

  L = symbchol();
  [LL, Ld, Lskip, Ladd] = blkchol(L, data.X);

  out_name = strrep(name, '_spd.mat', '_blkchol_oracle.mat');
  ordmmd_perm = L.perm;   % symbchol's own internal ordmmd call result is
  % not separately observable, but L.perm (post sfinit_ refinement) is
  % exactly what symfctmex/sfinit_ produced from it -- same value our
  % Python symbolic_cholesky() must reproduce.
  xsuper = L.xsuper;
  Lvals = LL;
  d = Ld;
  skip = full(Lskip);
  skip_ir = find(Lskip) - 1;   % 0-indexed column numbers, to match Python
  diagadd = full(Ladd);
  diagadd_ir = find(Ladd) - 1;

  save('-v7', fullfile(fixture_dir, out_name), ...
       'ordmmd_perm', 'xsuper', 'Lvals', 'd', 'skip_ir', 'skip', ...
       'diagadd_ir', 'diagadd');
  fprintf('%-20s n=%4d -> %s (nnz(L)=%d, nskip=%d, nadd=%d)\n', ...
          name, size(data.X, 1), out_name, nnz(LL), nnz(Lskip), nnz(Ladd));
end
