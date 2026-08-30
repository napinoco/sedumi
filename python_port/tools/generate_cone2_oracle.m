% generate_cone2_oracle.m
%
% Oracle fixtures for the second batch of Phase 3-b cone-math files:
% psdeig, psdfactor, psdinvscale, psdjmul, psdscale, qframeit, qinvjmul,
% qjmul, frameit, triumtriu.
%
% Reuses the exact same internal-format cone K as generate_cone_oracle.m
% (K.l=2, K.q=[4;3], K.s=[2;3], K.rsdpN=1: one real 2x2 and one complex
% Hermitian 3x3 PSD block) so K.N matches (31). The extra bookkeeping
% fields these files need (K.mainblks, K.qblkstart, K.sblkstart, K.lq)
% are normally computed by pretransfo.m from K.l/K.q/K.s alone (no free
% variables, no rotated cones, no split variables here, so pretransfo's
% formula reduces to a plain cumsum). The formulas used below were
% cross-checked line-by-line against a real pretransfo.m run on an
% equivalent (K.l, K.q, all-real K.s) problem -- confirmed to match
% exactly once pretransfo's own self-dual-variable augmentation (K.l+1,
% K.N+1) is subtracted back out. The complex-block doubling term
% (2*Ksc.^2) is taken directly from pretransfo.m's own source, which
% implements the same real/complex block-stacking convention already
% end-to-end verified for eigK/maxeigK/mineigK/eyeK in
% generate_cone_oracle.m.
%
% IMPORTANT non-obvious finding: eigK.m's own output `lab` INTERLEAVES
% each Lorentz block's [lo,hi] pair (lab = [L; lo_1,hi_1,lo_2,hi_2,...]),
% but qframeit.m/qinvjmul.m expect a *grouped* layout instead (lab =
% [L; lo_1,lo_2,...; hi_1,hi_2,...]) -- this is the layout trydif.m/
% widelen.m actually build by hand as `w.lab = [x(1:K.l).*z(1:K.l);
% detxz./lab2q; lab2q; psdeig(w.s,K)]` (lab2q = the "hi" group, computed
% directly, not the same code path as eigK.m at all). Passing eigK's own
% (interleaved) output to qframeit would silently compute something
% *different* from what real SeDuMi does. lab_grouped below is built by
% hand (lo_k, hi_k per Lorentz block) to match the real grouped
% convention, not derived from eigK's output.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cone2_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cone');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 44);

K.l = 2;
K.q = [4; 3];
K.s = [2; 3];
K.rsdpN = 1;
K.N = K.l + sum(K.q) + 2*2 + 2*(3*3);   % 31, same as generate_cone_oracle.m

lorN = length(K.q);
Ksr = K.s(1:K.rsdpN)'; Ksc = K.s(K.rsdpN+1:end)';
blkstart = cumsum([K.l+1, lorN, (K.q'-1), Ksr.^2, 2*Ksc.^2]);
K.mainblks  = blkstart([1, 2, 2+lorN]);
K.qblkstart = blkstart(2:2+lorN);
K.sblkstart = blkstart(2+lorN:end);
K.lq = K.mainblks(end) - 1;

xlen = K.N;
x = rand(xlen,1) - 0.5;
z = rand(xlen,1) - 0.5;

% Overwrite the PSD tail of x with genuinely positive-definite blocks
% (A*A' + n*I), since psdfactor/psdscale/psdinvscale need chol() to
% succeed -- a plain random vector's "PSD part" is essentially never PD.
psd_off = K.lq;   % PSD part starts right after the L+Q part (index lq+1, 1-indexed)
Ar = rand(2) - 0.5; Xr = Ar*Ar' + 2*eye(2);
x(psd_off+1 : psd_off+4) = Xr(:);
psd_off = psd_off + 4;
Ac = (rand(3) - 0.5) + 1i*(rand(3) - 0.5);
Xc = Ac*Ac' + 3*eye(3);
x(psd_off+1 : psd_off+9) = real(Xc(:));
x(psd_off+10 : psd_off+18) = imag(Xc(:));

% ---- Build the GROUPED Lorentz spectral-value layout by hand (see the
% note above -- this is NOT eigK(x,K)'s own interleaved output). ----
lo = zeros(lorN,1); hi = zeros(lorN,1);
xi = K.l + lorN;   % skip L-part and the lorN stacked x0's
tmp = sqrt(0.5);
for k = 1:lorN
    kk = K.q(k) - 1;
    x0 = x(K.l + k);
    nrm = norm(x(xi+1:xi+kk));
    lo(k) = tmp*(x0 - nrm);
    hi(k) = tmp*(x0 + nrm);
    xi = xi + kk;
end
psd_lab = eigK(x, K);
psd_lab = psd_lab(K.l+2*lorN+1:end);   % PSD tail: identical construction in eigK.m and w.lab
% L-part value is unused by qframeit/qinvjmul's own Lorentz-only slicing,
% but frameit.m does read lab(1:K.l) directly as its own L-part output.
lab_grouped = [x(1:K.l); lo; hi; psd_lab];

nq_vec = sum(K.q) - lorN;   % =5: total Lorentz "vector-part" length
frmq = rand(nq_vec,1) - 0.5;

% ---- psdeig (2-output form) ----
[lab_psd, q_psd] = psdeig(x, K);

% ---- psdfactor ----
[ux, ispos] = psdfactor(x, K);

% ---- psdinvscale ----
y_invscale = psdinvscale(ux, z, K);

% ---- psdjmul ----
z_jmul = psdjmul(x, z, K);

% ---- psdscale, transp=0 and transp=1, plain-vector ud ----
y_scale0 = psdscale(ux, z, K);
y_scale1 = psdscale(ux, z, K, 1);

% ---- psdscale, struct ud with a nontrivial permutation ----
% NOTE: psdscale.m reads perm PER BLOCK, ki entries at a time, as *local*
% 1..ki indices into that block (PP = perm(pi+1:pi+ki); XX(PP,PP) = XX) --
% not global 1..sum(K.s) indices. block1 (ki=2): [2,1]; block2 (ki=3): [2,3,1].
perm = [2; 1; 2; 3; 1];
ud_struct.u = ux;
ud_struct.perm = perm;
y_scale_perm0 = psdscale(ud_struct, z, K);
y_scale_perm1 = psdscale(ud_struct, z, K, 1);

% ---- triumtriu ----
z_triumtriu = triumtriu(x, z, K);

% ---- qframeit ----
x_qframeit = qframeit(lab_grouped, frmq, K);

% ---- qjmul (full internal-format vectors, matching wregion.m's own usage) ----
z_qjmul = qjmul(x, z, K);

% ---- qinvjmul ----
y_qinvjmul = qinvjmul(lab_grouped, frmq, z, K);

% ---- minpsdeig ----
mineig_psd = minpsdeig(x, K);

save('-v7', fullfile(out_dir, 'cone2.mat'), ...
     'x', 'z', 'K', ...
     'lab_grouped', 'frmq', ...
     'lab_psd', 'q_psd', 'ux', 'ispos', ...
     'y_invscale', 'z_jmul', 'y_scale0', 'y_scale1', ...
     'perm', 'y_scale_perm0', 'y_scale_perm1', ...
     'z_triumtriu', 'x_qframeit', 'z_qjmul', 'y_qinvjmul', 'mineig_psd');

fprintf('Cone2 oracle written to %s\n', out_dir);

% ---- frameit: separate, ALL-REAL PSD cone (cluster3-style) so a genuine
% qrK()-produced Householder frame (frms) can be used -- qrK/psdframeit
% are only verified for real-symmetric blocks (cluster 3), so this keeps
% frameit's own test decoupled from any complex-Hermitian-block question.
Kf.l = 2;
Kf.q = [4; 3];
Kf.s = [3; 2];
Kf.rsdpN = 2;
lenud_f = 3^2 + 2^2;
lorNf = length(Kf.q);
blkstart_f = cumsum([Kf.l+1, lorNf, (Kf.q'-1), Kf.s'.^2]);
Kf.mainblks = blkstart_f([1,2,2+lorNf]);
Kf.qblkstart = blkstart_f(2:2+lorNf);
Kf.sblkstart = blkstart_f(2+lorNf:end);
Kf.lq = Kf.mainblks(end) - 1;
Kf.N = Kf.l + sum(Kf.q) + sum(Kf.s.^2);

xf = rand(Kf.N,1) - 0.5;
lof = zeros(lorNf,1); hif = zeros(lorNf,1);
xi = Kf.l + lorNf;
for k = 1:lorNf
    kk = Kf.q(k) - 1;
    x0 = xf(Kf.l + k);
    nrm = norm(xf(xi+1:xi+kk));
    lof(k) = tmp*(x0 - nrm);
    hif(k) = tmp*(x0 + nrm);
    xi = xi + kk;
end
psd_labf = eigK(xf, Kf);
psd_labf = psd_labf(Kf.l+2*lorNf+1:end);
labf_grouped = [xf(1:Kf.l); lof; hif; psd_labf];

frmqf = rand(sum(Kf.q)-lorNf,1) - 0.5;
u_for_qrK = rand(lenud_f,1) - 0.5;
[frmsf, ~] = qrK(u_for_qrK, Kf);

x_frameit = frameit(labf_grouped, frmqf, frmsf, Kf);

save('-v7', fullfile(out_dir, 'cone2_frameit.mat'), ...
     'Kf', 'labf_grouped', 'frmqf', 'frmsf', 'x_frameit');

fprintf('Cone2 frameit oracle written to %s\n', out_dir);
