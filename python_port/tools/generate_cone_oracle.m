% generate_cone_oracle.m
%
% Runs the real eigK/maxeigK/mineigK/eyeK .m files (pure MATLAB/Octave,
% no MEX) against SeDuMi's INTERNAL cone-K representation (K.rsdpN
% present) -- the only representation sedumi.m's own iteration loop
% actually uses (confirmed by grep: eyeK is called exactly once in the
% whole codebase, from sdinit.m, always with this representation). The
% "external" (K.f/K.l/K.q/K.r/K.s, no rsdpN) branch in each of these
% files is ported too (see cone.py) but not verified here.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cone_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cone');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 44);

% Internal-format cone: K.l=2 LP, K.q=[4,3] two Lorentz blocks (stacked
% x0-first per pretransfo.m's convention), K.s=[2,3] one real 2x2 PSD
% block and one complex Hermitian 3x3 PSD block (rsdpN=1 marks the
% boundary).
K.l = 2;
K.q = [4; 3];
K.s = [2; 3];
K.rsdpN = 1;
K.N = K.l + sum(K.q) + 2*2 + 2*(3*3);  % LP + Lorentz(full, incl. x0) + real 2x2 + complex 3x3 (re+im)

nq_x0 = length(K.q);          % internal format: x0's stacked first
nq_vec = sum(K.q) - nq_x0;    % remaining Lorentz vector entries
n_real_sdp = 2*2;             % K.s(1)=2, real
n_cplx_sdp = 2*(3*3);         % K.s(2)=3, complex (re+im)
xlen = K.l + nq_x0 + nq_vec + n_real_sdp + n_cplx_sdp;

x = rand(xlen, 1) - 0.5;
% Make the PSD blocks' "diagonal-ish" entries positive-biased so eigK
% exercises a mix of positive/negative eigenvalues realistically (not
% required for correctness, just a saner test vector).

lab = eigK(x, K);
labmax = maxeigK(x, K);
labmin = mineigK(x, K);
idK = eyeK(K);

Kl = K.l; Kq = K.q; Ks = K.s; KrsdpN = K.rsdpN; KN = K.N;
save('-v7', fullfile(out_dir, 'eigK.mat'), ...
     'x', 'Kl', 'Kq', 'Ks', 'KrsdpN', 'KN', 'lab', 'labmax', 'labmin', 'idK');

fprintf('Cone oracle written to %s\n', out_dir);
