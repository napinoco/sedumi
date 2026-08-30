% generate_ordmmd_oracle.m
%
% Runs the REAL Octave/MEX ordmmdmex() (compiled from the same ordmmd.c
% used by the Python bindings, but through the original MATLAB/Octave
% MEX calling convention) on each fixture matrix in
% tests/fixtures/ordmmd/*.mat, and saves the resulting permutation next
% to it. This is the oracle python_port/tests/test_ordmmd.py checks the
% ctypes ordmmd() binding against -- not "a valid minimum degree
% ordering" in the abstract, but "bit-for-bit the same ordering the MEX
% build produces for the same input", since ordmmd_ itself is completely
% unchanged by the port (only the calling convention is different).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_ordmmd_oracle"

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
  perm = ordmmdmex(data.A);
  out_name = strrep(name, '.mat', '_oracle.mat');
  save('-v7', fullfile(fixture_dir, out_name), 'perm');
  fprintf('%-20s n=%4d -> %s (perm(1)=%d perm(end)=%d)\n', ...
          name, size(data.A, 1), out_name, perm(1), perm(end));
end
