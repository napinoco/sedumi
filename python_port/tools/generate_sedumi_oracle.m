% generate_sedumi_oracle.m
%
% Runs the real Octave sedumi.m end-to-end on small LP and SOCP (Lorentz)
% problems, for Phase 3-d's golden-reference verification of this port's
% sedumi.py driver. Saves the original problem data (At,b,c,K) plus the
% solution (x,y,info) so the Python side can both re-solve and sanity-
% check the saved solution's feasibility independently.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_sedumi_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sedumi');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

pars.fid = 0;   % quiet

function save_case(out_dir, name, At, b, c, K, pars)
    [x, y, info] = sedumi(At, b, c, K, pars);
    save('-v7', fullfile(out_dir, [name '.mat']), 'At', 'b', 'c', 'K', 'pars', 'x', 'y', 'info');
    fprintf('  %s: iter=%d numerr=%d pinf=%d dinf=%d cx=%g by=%g\n', ...
        name, info.iter, info.numerr, info.pinf, info.dinf, full(sum(c.*x)), full(sum(b.*y)));
end

% Case 1: small feasible LP (K.l only), At in SeDuMi's native N x m form.
rand('seed', 501); randn('seed', 501);
n = 8; m = 5;
K = struct('f', 0, 'l', n, 'q', zeros(1,0), 'r', zeros(1,0), 's', zeros(1,0));
xfeas = rand(n,1) + 0.1;     % strictly positive
At = rand(n,m) - 0.5;
b = At' * xfeas;              % guarantees primal feasibility
c = rand(n,1) - 0.5;
save_case(out_dir, 'lp_feasible', At, b, c, K, pars);

% Case 2: LP feasibility problem with a free block (K.f>0).
rand('seed', 502); randn('seed', 502);
nf = 2; nl = 6; m = 5;
n = nf + nl;
K = struct('f', nf, 'l', nl, 'q', zeros(1,0), 'r', zeros(1,0), 's', zeros(1,0));
xfeas = [randn(nf,1); rand(nl,1)+0.1];
At = rand(n,m) - 0.5;
b = At' * xfeas;
c = rand(n,1) - 0.5;
save_case(out_dir, 'lp_free_feasible', At, b, c, K, pars);

% Case 3: SOCP (single Lorentz cone) feasible problem.
rand('seed', 503); randn('seed', 503);
nl = 2; nq = 4; m = 4;
n = nl + nq;
K = struct('f', 0, 'l', nl, 'q', nq, 'r', zeros(1,0), 's', zeros(1,0));
xq = randn(nq-1,1);
xfeas = [rand(nl,1)+0.1; norm(xq)+0.5; xq];   % strictly interior to the Lorentz cone
At = rand(n,m) - 0.5;
b = At' * xfeas;
c = rand(n,1) - 0.5;
save_case(out_dir, 'socp_feasible', At, b, c, K, pars);

% Case 4: two Lorentz cones, no linear part.
rand('seed', 504); randn('seed', 504);
nq1 = 3; nq2 = 3; m = 5;
n = nq1 + nq2;
K = struct('f', 0, 'l', 0, 'q', [nq1, nq2], 'r', zeros(1,0), 's', zeros(1,0));
x1 = randn(nq1-1,1); x2 = randn(nq2-1,1);
xfeas = [norm(x1)+0.5; x1; norm(x2)+0.5; x2];
At = rand(n,m) - 0.5;
b = At' * xfeas;
c = rand(n,1) - 0.5;
save_case(out_dir, 'socp_two_blocks', At, b, c, K, pars);

% Case 5: primal infeasible LP (deliberately inconsistent Ax=b with x>=0).
rand('seed', 505); randn('seed', 505);
n = 5; m = 3;
K = struct('f', 0, 'l', n, 'q', zeros(1,0), 'r', zeros(1,0), 's', zeros(1,0));
At = rand(n,m) - 0.5;
b = [1; 1; 1] + sum(abs(At),1)';   % b_i far larger than any achievable A'x with x>=0 in the wrong direction
b = -abs(b);                        % push toward infeasibility for x>=0
c = rand(n,1) - 0.5;
save_case(out_dir, 'lp_infeasible', At, b, c, K, pars);

% Case 6: rotated Lorentz cone (K.r), feasible.
rand('seed', 506); randn('seed', 506);
nl = 2; nr = 4; m = 4;
n = nl + nr;
K = struct('f', 0, 'l', nl, 'q', zeros(1,0), 'r', nr, 's', zeros(1,0));
xr = randn(nr-2,1);
xfeas = [rand(nl,1)+0.1; 1+rand(1); 1+rand(1); xr];   % 2*x1*x2 >= norm(xr)^2 with margin
At = rand(n,m) - 0.5;
b = At' * xfeas;
c = rand(n,1) - 0.5;
save_case(out_dir, 'rotated_cone_feasible', At, b, c, K, pars);

% Case 7: larger, sparser LP -- exercises symbchol's ordmmd (non-fully-
% dense) branch, unlike cases 1-6's small dense-A problems. Seed chosen
% (out of a small sweep) to land on a clean primal-dual optimal result
% (pinf=dinf=0), since a rejected-by-residual-check (x0 forced to 0)
% outcome leaves the returned y numerically under-determined (only x's
% Farkas-direction scaling gets fixed up in that branch) and isn't a
% meaningful test of exact agreement.
rand('seed', 600); randn('seed', 600);
n = 25; m = 10;
K = struct('f', 0, 'l', n, 'q', zeros(1,0), 'r', zeros(1,0), 's', zeros(1,0));
xfeas = rand(n,1) + 0.1;
At = sprand(n, m, 0.3);
b = full(At' * xfeas);
c = rand(n,1) - 0.5;
save_case(out_dir, 'lp_sparse_feasible', At, b, c, K, pars);

fprintf('sedumi oracle written to %s\n', out_dir);
