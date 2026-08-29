% generate_golden.m
%
% Phase 0 (verification baseline) for the SeDuMi -> Python port.
%
% Runs the current Octave/MEX build of SeDuMi on every example problem in
% examples/*.mat and saves the full solution (x, y, info, and the derived
% objective values) as a "golden" reference. Every later phase of the port
% (C kernel extraction, Python translation, ...) is checked against these
% files so that a numerical regression is caught immediately.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_golden"

pkg_root = fileparts(fileparts(mfilename('fullpath')));  % python_port/
repo_root = fileparts(pkg_root);                          % repo root
example_path = fullfile(repo_root, 'examples');
golden_dir = fullfile(pkg_root, 'tests', 'golden');

addpath(repo_root);
addpath(fullfile(repo_root, 'conversion'));

if ~exist(golden_dir, 'dir')
  mkdir(golden_dir);
end

test_problems = { ...
  'arch0.mat',                  -5.665170e-01; ...
  'control07.mat',              -2.062510e+01; ...
  'nb.mat',                     -5.070309e-02; ...
  'OH_2Pi_STO-6GN9r12g1T2.mat',  7.946708e+01; ...
  'trto3.mat',                  -1.279999e+04; ...
  'quantum.mat',                -0.75395345};

summary = struct('name', {}, 'cx', {}, 'by', {}, 'expected', {}, ...
                  'cx_relerr', {}, 'by_relerr', {}, 'iter', {}, ...
                  'numerr', {}, 'pinf', {}, 'dinf', {}, 'err', {});

for i = 1:size(test_problems, 1)
  name = test_problems{i, 1};
  expected = test_problems{i, 2};
  fprintf('\n=== [%d/%d] %s ===\n', i, size(test_problems, 1), name);

  data = load(fullfile(example_path, name), 'At', 'b', 'c', 'K');

  pars = struct();
  pars.errors = 1;
  pars.fid = 1;

  [x, y, info] = sedumi(data.At, data.b, data.c, data.K, pars);

  cx = data.c' * x;
  by = data.b' * y;
  cx_relerr = abs(cx - expected) / abs(expected);
  by_relerr = abs(by - expected) / abs(expected);

  fprintf('  cx = %.10e (relerr %.2e)\n', cx, cx_relerr);
  fprintf('  by = %.10e (relerr %.2e)\n', by, by_relerr);
  fprintf('  iter = %d, numerr = %d, pinf = %d, dinf = %d\n', ...
          info.iter, info.numerr, info.pinf, info.dinf);

  out_file = fullfile(golden_dir, strrep(name, '.mat', '_golden.mat'));
  save('-v7', out_file, 'x', 'y', 'info', 'cx', 'by', 'expected', ...
       'cx_relerr', 'by_relerr');

  % info.err (the DIMACS error vector) is only populated by pars.errors=1
  % for real-valued problems; SeDuMi does not compute it for problems with
  % complex data (e.g. quantum.mat), so guard against the missing field.
  if isfield(info, 'err')
    err_vec = info.err;
  else
    err_vec = nan(1, 6);
  end

  summary(end+1) = struct('name', name, 'cx', cx, 'by', by, ...
    'expected', expected, 'cx_relerr', cx_relerr, 'by_relerr', by_relerr, ...
    'iter', info.iter, 'numerr', info.numerr, 'pinf', info.pinf, ...
    'dinf', info.dinf, 'err', err_vec);
end

save('-v7', fullfile(golden_dir, 'summary.mat'), 'summary');

fprintf('\n%s\nSummary\n%s\n', repmat('-', 1, 60), repmat('-', 1, 60));
for i = 1:numel(summary)
  s = summary(i);
  fprintf('%-32s cx_relerr=%.2e by_relerr=%.2e iter=%3d numerr=%d\n', ...
          s.name, s.cx_relerr, s.by_relerr, s.iter, s.numerr);
end
fprintf('\nGolden reference files written to: %s\n', golden_dir);
