% generate_pretransfo_oracle.m
%
% Runs the real Octave pretransfo.m on a battery of small problems, one
% per cone-transformation kind pretransfo.m implements, and saves
% inputs+outputs so python_port/tests/test_pretransfo.py can check the
% Python port against them without needing Octave installed to run.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_pretransfo_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'pretransfo');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

pars.free = 0; pars.sdp = 1; pars.errors = 0;

function save_case(out_dir, name, At, b, c, K, pars)
    [At2, b2, c2, K2, prep] = pretransfo(At, b, c, K, pars);
    save('-v7', fullfile(out_dir, [name '.mat']), ...
         'At', 'b', 'c', 'K', 'pars', 'At2', 'b2', 'c2', 'K2', 'prep');
    fprintf('  %s: K2.N=%d K2.l=%d K2.q=[%s] K2.s=[%s] K2.rsdpN=%d\n', ...
        name, K2.N, K2.l, sprintf('%d ',K2.q), sprintf('%d ',K2.s), K2.rsdpN);
end

rand('seed', 123);

%% Case 1: pure LP
K = struct('f',0,'l',5,'q',zeros(1,0),'r',zeros(1,0),'s',zeros(1,0));
n = K.l;
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'lp_only', At, b, c, K, pars);

%% Case 2: LP + free variables, split (pars.free=0)
K = struct('f',2,'l',3,'q',zeros(1,0),'r',zeros(1,0),'s',zeros(1,0));
n = K.f + K.l;
At = rand(n,4)-0.5; b = rand(4,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'free_split', At, b, c, K, pars);

%% Case 3: LP + free variables, into Lorentz cone (pars.free=1)
pars2 = pars; pars2.free = 1;
save_case(out_dir, 'free_lorentz', At, b, c, K, pars2);

%% Case 4: LP + Lorentz cones (several blocks)
K = struct('f',0,'l',2,'q',[3;4],'r',zeros(1,0),'s',zeros(1,0));
n = K.l + sum(K.q);
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'lorentz', At, b, c, K, pars);

%% Case 5: rotated Lorentz cones
K = struct('f',0,'l',1,'q',zeros(1,0),'r',[3;4],'s',zeros(1,0));
n = K.l + sum(K.r);
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'rotated_lorentz', At, b, c, K, pars);

%% Case 6: mixed L + Lorentz + rotated Lorentz
K = struct('f',0,'l',2,'q',[3;4],'r',[3;5],'s',zeros(1,0));
n = K.l + sum(K.q) + sum(K.r);
At = rand(n,4)-0.5; b = rand(4,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'mixed_l_q_r', At, b, c, K, pars);

%% Case 7: real PSD blocks (no diagonal, no complex)
K = struct('f',0,'l',1,'q',zeros(1,0),'r',zeros(1,0),'s',[3;2]);
n = K.l + sum(K.s.^2);
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'real_sdp', At, b, c, K, pars);

%% Case 8: real PSD with one diagonal 1x1 block (auto-flagged) and one
%% *structurally* diagonal 3x3 block (all off-diagonal entries exactly
%% zero in both At and c).
K = struct('f',0,'l',0,'q',zeros(1,0),'r',zeros(1,0),'s',[1;3;2]);
n = sum(K.s.^2);
At = rand(n,3)-0.5; c = rand(n,1)-0.5;
% zero out off-diagonal entries of the 3x3 block (positions 2:10, local 3x3 offset=1 (0-idx) after the 1x1 block)
off = K.s(1)^2;  % = 1
D = reshape(1:9,3,3); offdiag = D(~eye(3));
At(off+offdiag, :) = 0;
c(off+offdiag) = 0;
b = rand(3,1)-0.5;
save_case(out_dir, 'real_sdp_diag', At, b, c, K, pars);

%% Case 9: single complex Hermitian PSD block
K = struct('f',0,'l',0,'q',zeros(1,0),'r',zeros(1,0),'s',3);
K.scomplex = 1;
n = K.s^2;
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'complex_sdp_single', At, b, c, K, pars);

%% Case 10: all-complex PSD blocks (2 blocks, both complex)
K = struct('f',0,'l',0,'q',zeros(1,0),'r',zeros(1,0),'s',[2;3]);
K.scomplex = [1,2];
n = sum(K.s.^2);
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'complex_sdp_all', At, b, c, K, pars);

% NOTE: a case mixing one real and one complex SDP block (K.s=[2;3],
% K.scomplex=2, marking only the SECOND block complex) is deliberately
% NOT included here: it triggers a genuine bug in pretransfo.m itself
% (see pretransfo.py's module docstring and
% test_scomplex_partial_selection_not_octave_bug_compatible) where
% Octave's own output would not be valid ground truth for a corrected
% port.

%% Case 12: K.ycomplex (complex-valued dual variables)
K = struct('f',0,'l',3,'q',zeros(1,0),'r',zeros(1,0),'s',zeros(1,0));
K.ycomplex = [1;3];
n = K.l;
At = (rand(n,3)-0.5) + 1i*(rand(n,3)-0.5);
b = (rand(3,1)-0.5) + 1i*(rand(3,1)-0.5);
c = rand(n,1)-0.5;  % c itself must stay consistent with L-part realness expectations used elsewhere; keep real
save_case(out_dir, 'ycomplex', At, b, c, K, pars);

%% Case 13: K.xcomplex marking a free variable's imaginary part, plus a
%% Lorentz-cone trace-part complex marking.
K = struct('f',2,'l',0,'q',[4],'r',zeros(1,0),'s',zeros(1,0));
K.xcomplex = [1, 3];  % 1st free var, and 1st Lorentz block's trace part (position K.f+1=3)
n = K.f + sum(K.q);
At = rand(n,3)-0.5; b = rand(3,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'xcomplex_mixed', At, b, c, K, pars);

%% Case 14: everything at once -- L, Lorentz, rotated Lorentz, real SDP,
%% free vars (both split and Lorentz variants). A SINGLE SDP block is used
%% (not a complex one) deliberately: K.scomplex marking exactly one of
%% *several* SDP blocks complex hits a real upstream pretransfo.m bug
%% (see pretransfo.py's module docstring) where Octave's own output would
%% not be valid ground truth to test a corrected port against.
K = struct('f',2,'l',2,'q',[3],'r',[3],'s',[2]);
n = K.f + K.l + sum(K.q) + sum(K.r) + sum(K.s.^2);
At = rand(n,4)-0.5; b = rand(4,1)-0.5; c = rand(n,1)-0.5;
save_case(out_dir, 'kitchen_sink_split', At, b, c, K, pars);
save_case(out_dir, 'kitchen_sink_lorentz', At, b, c, K, pars2);

fprintf('Pretransfo oracle written to %s\n', out_dir);
