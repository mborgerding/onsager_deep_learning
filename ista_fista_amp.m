% simulation parameters

% if you do not have these .mat files, run:
%       python save_problem.py      
% (requires: python, tensorflow, scipy )
load problem_Giid.mat;disp('loaded Gaussian A problem')
%load problem_k15.mat;disp('loaded kappa=15 problem')
L = size(x,2);
[M,N] = size(A);
Afro2 = norm(A,'fro')^2;

assert(abs(Afro2-N)/N < 0.01,'I was assuming this would have unit norm columns')
z = A*x;
wvar =  mean((y(:)-z(:)).^2);
SNRdB_test = 10*log10(mean(abs(z(:)).^2)/wvar);
supp = x(:)~=0;
K = mean(supp)*N;

% algorithm parameters
T = 1e2; % AMP iterations
Ti = 1e3; % FISTA iterations
Tii = 1e4; % ISTA iterations
alf = 1.1402; % amp tuning parameter [1.1402]
nmse_dB_report = -35;
eta = @(r,lam) sign(r).*max(bsxfun(@minus,abs(r),lam),0); 
tqq = 10; tqqi = 100; tqqii = 1000; % iteration for qqplot 

% run AMP
Bmf = A'; % matched filter 
xhat = zeros(N,L); % initialization of signal estimate
v = zeros(M,L); % initialization of residual
nmse_amp = [ones(1,L);zeros(T,L)];
qq = true;
report = true;
for t=1:T
  g = (N/M)*mean(xhat~=0,1); % onsager gain
  v = y - A*xhat + bsxfun(@times,v,g); % residual
  rhat = xhat + Bmf*v; % denoiser input
  rvar = sum(abs(v).^2,1)/M; % denoiser input err var
  xhat = eta(rhat, alf*sqrt(rvar)); % estimate
  nmse_amp(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
  if qq&(mean(nmse_amp(t+1,:))<0.1), 
    figure(2)
    subplot(131); 
    qqplot(rhat(:,1)-x(:,1)); 
    axis('square')
    title(['AMP at iteration ',num2str(t)]); 
    drawnow;
    qq = false;
  end
  if report&&(mean(nmse_amp(t+1,:))<10^(nmse_dB_report/10)), 
    fprintf('AMP reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
    report = false;
  end
end
if mean(nmse_amp(end,:)) > .1
    lam_mf = .01;
    fprintf('AMP did not converge! ... using a wild guess lam_mf=%f for ISTA,FISTA\n',lam_mf);
else
    xhat_mf = xhat;
    fprintf('AMP terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_amp(end,:))) );
    lam_mf = alf*sqrt(sum(abs(v).^2,1)/M).*(1-sum(xhat~=0,1)/M); % lam for lasso
    %lam_mf = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso
end

% run FISTA
scale = .999/norm(Bmf*A);
B = scale*Bmf;
xhat = zeros(N,L); % initialization of signal estimate
xhat_old = zeros(N,L);
nmse_fista = [ones(1,L);zeros(Ti,L)];
qq = true;
report = true;
for t=1:Ti
  v = y - A*xhat; % residual
  rhat = xhat + B*v + ((t-2)/(t+1))*(xhat-xhat_old); % denoiser input
  xhat_old = xhat;
  xhat = eta(rhat, lam_mf*scale); % estimate
  nmse_fista(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
  if qq&&(mean(nmse_fista(t+1,:))<0.1), 
    figure(2)
    subplot(132); 
    qqplot(rhat(:,1)-x(:,1)); 
    axis('square')
    title(['FISTA at iteration ',num2str(t)]); 
    drawnow;
    qq = false;
  end
  if report&&(mean(nmse_fista(t+1,:))<10^(nmse_dB_report/10)), 
    fprintf('FISTA reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
    report = false;
  end
end
xhat_fista_mf = xhat;
fprintf('FISTA terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_fista(end,:))) );
lam_mf_test = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso

% run ISTA
xhat = zeros(N,L); % initialization of signal estimate
nmse_ista = [ones(1,L);zeros(Tii,L)];
qq = true;
report = true;
for t=1:Tii
  v = y - A*xhat; % residual
  rhat = xhat + B*v; % denoiser input
  xhat = eta(rhat, lam_mf*scale); % estimate
  nmse_ista(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
  if qq&&(mean(nmse_ista(t+1,:))<0.1), 
    figure(2)
    subplot(133); 
    qqplot(rhat(:,1)-x(:,1)); 
    axis('square')
    title(['ISTA at iteration ',num2str(t)]); 
    drawnow;
    qq = false;
  end
  if report&&(mean(nmse_ista(t+1,:))<10^(nmse_dB_report/10)), 
    fprintf('ISTA reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
    report = false;
  end
end
xhat_ista_mf = xhat;
fprintf('ISTA terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_ista(end,:))) );
lam_mf_test = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso

% plot results
figure(1)
handy = semilogx([0:Tii],10*log10(mean(nmse_ista,2)),'b.-',...
         [0:Ti],10*log10(mean(nmse_fista,2)),'g.-',...
         [0:T],10*log10(mean(nmse_amp,2)),'k.-');
set(handy(2),'Color',[0 0.5 0])
legend('ISTA','FISTA','AMP')
ylabel('NMSE [dB]')
xlabel('iterations')
grid on
title(['N=',num2str(N),', M=',num2str(M),', E[K]=',num2str(K),', SNRdB=',num2str(SNRdB_test),', MMV=',num2str(L)])
