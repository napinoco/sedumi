% search_blkchol_edge_cases.m
%
% Runs the real Octave/MEX symbchol()+blkchol() on every candidate matrix
% in tests/fixtures/blkchol_edge_candidates/*.mat and reports nskip/nadd,
% to find which candidates actually exercise blkchol's pivot-skip and
% diagonal-stabilization paths (see search_blkchol_edge_cases.py).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; search_blkchol_edge_cases"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
cand_dir = fullfile(pkg_root, 'tests', 'fixtures', 'blkchol_edge_candidates');

addpath(repo_root);
global ADA_sedumi_

files = dir(fullfile(cand_dir, '*.mat'));
for i = 1:numel(files)
  name = files(i).name;
  data = load(fullfile(cand_dir, name), 'X');
  ADA_sedumi_ = data.X;
  try
    L = symbchol();
    [LL, Ld, Lskip, Ladd] = blkchol(L, data.X);
    nskip = nnz(Lskip);
    nadd = nnz(Ladd);
    mineig_diag = min(diag(data.X));
    fprintf('%-30s nskip=%3d nadd=%3d mindiag=%.4f\n', name, nskip, nadd, mineig_diag);
  catch err
    fprintf('%-30s ERROR: %s\n', name, err.message);
  end
end
