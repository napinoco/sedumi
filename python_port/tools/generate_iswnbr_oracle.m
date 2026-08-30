% generate_iswnbr_oracle.m
%
% Runs the real Octave/MEX iswnbr.c across scenarios covering both of
% its branches (0<theta<1 and the theta==1 special case) and its
% "inconclusive set Q nonempty" path.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_iswnbr_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'iswnbr');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 55);

%% Case 1: 0 < theta < 1, moderate spread (large n so Q is nontrivial)
w1 = rand(20,1) * 5 + 0.01;
theta1 = 0.25^2;
[delta1, h1, alpha1] = iswnbr(w1, theta1);
save('-v7', fullfile(out_dir, 'case1.mat'), 'w1', 'theta1', 'delta1', 'h1', 'alpha1');

%% Case 2: theta == 1 special case
w2 = rand(15,1) * 3 + 0.01;
theta2 = 1.0;
[delta2, h2, alpha2] = iswnbr(w2, theta2);
save('-v7', fullfile(out_dir, 'case2.mat'), 'w2', 'theta2', 'delta2', 'h2', 'alpha2');

%% Case 3: a very wide spread of values (large ratio max/min) to stress
%% the T/Q partitioning logic with several values ending up in T.
w3 = [0.001; 0.01; 0.1; 1; 10; 100; 1000; rand(10,1)*50 + 0.5];
theta3 = 0.5^2;
[delta3, h3, alpha3] = iswnbr(w3, theta3);
save('-v7', fullfile(out_dir, 'case3.mat'), 'w3', 'theta3', 'delta3', 'h3', 'alpha3');

%% Case 4: degenerate -- one entry <= 0 (infeasible point)
w4 = rand(10,1) + 0.1;
w4(3) = -0.5;
theta4 = 0.25^2;
[delta4, h4, alpha4] = iswnbr(w4, theta4);
save('-v7', fullfile(out_dir, 'case4.mat'), 'w4', 'theta4', 'delta4', 'h4', 'alpha4');

fprintf('iswnbr oracle written to %s\n', out_dir);
