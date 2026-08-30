pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
cand_dir = fullfile(pkg_root, 'tests', 'fixtures', 'blkchol_diagadd_candidates2');

addpath(repo_root);
global ADA_sedumi_

files = dir(fullfile(cand_dir, '*.mat'));
nfound = 0;
for i = 1:numel(files)
  name = files(i).name;
  data = load(fullfile(cand_dir, name), 'X');
  ADA_sedumi_ = data.X;
  try
    L = symbchol();
    [LL, Ld, Lskip, Ladd] = blkchol(L, data.X);
    nadd = nnz(Ladd);
    if nadd > 0
      fprintf('%-20s nskip=%3d nadd=%3d  <-- FOUND\n', name, nnz(Lskip), nadd);
      nfound = nfound + 1;
    end
  catch err
    fprintf('%-20s ERROR: %s\n', name, err.message);
  end
end
fprintf('\n%d/%d candidates with nadd>0\n', nfound, numel(files));
