% generate_blkchol_edge_oracle.m
%
% Same as generate_blkchol_oracle.m, but for the hand-selected edge-case
% fixtures in tests/fixtures/blkchol_edge/*.mat -- matrices specifically
% chosen (via python_port/tests/fixtures/search_blkchol_diagadd2.py and
% a random search logged in the commit history) to exercise blkchol's
% pivot-skip (nskip>0) and diagonal-stabilization (nadd>0) paths, which
% none of the well-conditioned fixtures in tests/fixtures/blkchol/
% trigger.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_blkchol_edge_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
fixture_dir = fullfile(pkg_root, 'tests', 'fixtures', 'blkchol_edge');

addpath(repo_root);
global ADA_sedumi_

files = dir(fullfile(fixture_dir, '*.mat'));
for i = 1:numel(files)
  name = files(i).name;
  if ~isempty(strfind(name, '_oracle'))
    continue;
  end
  path = fullfile(fixture_dir, name);
  data = load(path, 'X');
  ADA_sedumi_ = data.X;

  L = symbchol();
  [LL, Ld, Lskip, Ladd] = blkchol(L, data.X);

  out_name = strrep(name, '.mat', '_oracle.mat');
  ordmmd_perm = L.perm;
  xsuper = L.xsuper;
  Lvals = LL;
  d = Ld;
  skip_ir = find(Lskip) - 1;
  skip_val = full(Lskip(find(Lskip)));
  diagadd_ir = find(Ladd) - 1;
  diagadd_val = full(Ladd(find(Ladd)));

  save('-v7', fullfile(fixture_dir, out_name), ...
       'ordmmd_perm', 'xsuper', 'Lvals', 'd', ...
       'skip_ir', 'skip_val', 'diagadd_ir', 'diagadd_val');
  fprintf('%-20s n=%3d -> %s (nskip=%d, nadd=%d)\n', ...
          name, size(data.X, 1), out_name, nnz(Lskip), nnz(Ladd));
end
