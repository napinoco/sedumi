% generate_symfct_oracle.m
%
% Runs the REAL Octave/MEX symfctmex(X, perm) -- with perm taken from the
% real ordmmdmex(X), exactly the sequence symbchol.m uses -- on each
% fixture matrix in tests/fixtures/ordmmd/*.mat, and saves L.L (pattern),
% L.perm, L.xsuper next to it. This is the oracle
% python_port/tests/test_symfct.py checks the ctypes symbolic_cholesky()
% binding against.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_symfct_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
fixture_dir = fullfile(pkg_root, 'tests', 'fixtures', 'ordmmd');

addpath(repo_root);

files = dir(fullfile(fixture_dir, '*.mat'));
for i = 1:numel(files)
  name = files(i).name;
  if ~isempty(strfind(name, '_oracle'))
    continue;
  end
  path = fullfile(fixture_dir, name);
  data = load(path, 'A');
  ordmmd_perm = ordmmdmex(data.A);
  L = symfctmex(data.A, ordmmd_perm);
  out_name = strrep(name, '.mat', '_symfct_oracle.mat');
  Lpattern = spones(L.L);
  perm = L.perm;      % the (possibly sfinit_-refined) FINAL permutation
  xsuper = L.xsuper;
  save('-v7', fullfile(fixture_dir, out_name), ...
       'Lpattern', 'perm', 'xsuper', 'ordmmd_perm');
  fprintf('%-20s n=%4d -> %s (nnz(L)=%d, nsuper=%d)\n', ...
          name, size(data.A, 1), out_name, nnz(L.L), numel(L.xsuper) - 1);
end
