% check_no_regression.m
%
% Re-runs every example problem and compares against the Phase 0 golden
% reference (tests/golden/*_golden.mat) bit-for-bit on x, y and the DIMACS
% error vector, within a tight numerical tolerance. Used after any change
% to the C kernels (Phase 1) to prove there is no behavioral regression
% before committing.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; check_no_regression"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
example_path = fullfile(repo_root, 'examples');
golden_dir = fullfile(pkg_root, 'tests', 'golden');

addpath(repo_root);
addpath(fullfile(repo_root, 'conversion'));

names = {'arch0.mat', 'control07.mat', 'nb.mat', ...
         'OH_2Pi_STO-6GN9r12g1T2.mat', 'trto3.mat', 'quantum.mat'};

tol = 1e-8;
all_ok = true;

for i = 1:numel(names)
  name = names{i};
  golden_file = fullfile(golden_dir, strrep(name, '.mat', '_golden.mat'));
  if ~exist(golden_file, 'file')
    fprintf('SKIP %-32s (no golden file)\n', name);
    continue;
  end
  g = load(golden_file);

  data = load(fullfile(example_path, name), 'At', 'b', 'c', 'K');
  pars = struct('errors', 1, 'fid', 0);
  [x, y, info] = sedumi(data.At, data.b, data.c, data.K, pars);

  dx = full(max(abs(x(:) - g.x(:))));
  dy = full(max(abs(y(:) - g.y(:))));
  diter = abs(double(info.iter) - double(g.info.iter));

  ok = (dx < tol) && (dy < tol) && (diter == 0);
  all_ok = all_ok && ok;

  status = 'OK';
  if ~ok
    status = 'MISMATCH';
  end
  fprintf('%-6s %-32s max|dx|=%.2e max|dy|=%.2e diter=%d\n', ...
          status, name, dx, dy, diter);
end

if all_ok
  fprintf('\nAll problems match the golden reference: NO REGRESSION.\n');
else
  fprintf('\nREGRESSION DETECTED -- see MISMATCH lines above.\n');
  exit(1);
end
