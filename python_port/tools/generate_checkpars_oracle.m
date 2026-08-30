% generate_checkpars_oracle.m
%
% Runs the real Octave checkpars.m on a few input scenarios (empty pars,
% partial pars, out-of-range values needing clamping, nested chol/cg
% overrides) and saves the resulting filled-in pars structs.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_checkpars_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'checkpars');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

pars_empty = struct();
out_empty = checkpars(pars_empty);
save('-v7', fullfile(out_dir, 'empty.mat'), 'out_empty');

pars_partial = struct('beta', 0.95, 'theta', 0.001, 'alg', 5, 'eps', 1e-10);
out_partial = checkpars(pars_partial);
save('-v7', fullfile(out_dir, 'partial.mat'), 'pars_partial', 'out_partial');

pars_w = struct('w', [2.5, 1e-10]);
out_w = checkpars(pars_w);
save('-v7', fullfile(out_dir, 'w.mat'), 'pars_w', 'out_w');

pars_w_bad = struct('w', [1,2,3]);
out_w_bad = checkpars(pars_w_bad);
save('-v7', fullfile(out_dir, 'w_bad.mat'), 'pars_w_bad', 'out_w_bad');

pars_nested = struct('chol', struct('skip', 0), 'cg', struct('maxiter', 10));
out_nested = checkpars(pars_nested);
save('-v7', fullfile(out_dir, 'nested.mat'), 'pars_nested', 'out_nested');

pars_alg0 = struct('alg', 0);
out_alg0 = checkpars(pars_alg0);
save('-v7', fullfile(out_dir, 'alg0.mat'), 'pars_alg0', 'out_alg0');

fprintf('Checkpars oracle written to %s\n', out_dir);
