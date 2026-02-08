function parameter = adaptation_parameter(target_population_normalized_all,target_fitness_all,...
    source_population_normalized_all,source_fitness_all,solution_unadapted,target_opt,source_opt,prior_available,prior_uncertainty)

target_population_normalized = target_population_normalized_all{end};
target_fitness = target_fitness_all{end};
source_population_normalized = source_population_normalized_all{end};
source_fitness = source_fitness_all{end};
[popsize,dim] = size(target_population_normalized);

if prior_available
    methods = {'M1-Tp','M2-A','OC-K','Direct','Prior'}; % ,'ROC-L'
else
    methods = {'M1-Tp','M2-A','OC-K','Direct'};
end
p = ones(1,length(methods))*1/length(methods);
    

cum_p = cumsum(p);
r = rand();
for i = 1:length(cum_p)
    if r < cum_p(i)
        idx_method = i;
        break;
    end
end
% idx_method = randperm(length(methods));
method = methods{idx_method};
switch(method)
    case 'M1-Tp'
        percentage = 0.4;
        num_estimate = ceil(percentage*popsize);
        [~,idxs] = sort(source_fitness);
        moment1_source = mean(source_population_normalized(idxs(1:num_estimate),:));
        [~,idxt] = sort(target_fitness);
        moment1_target = mean(target_population_normalized(idxt(1:num_estimate),:));
        solution_adapted_normalized = solution_unadapted+(moment1_target-moment1_source);
    case 'M1-Tr'
        num_front = 5;
        [~,idxs] = sort(source_fitness);
        moment1_source = source_population_normalized(idxs(randi(num_front)),:);
        [~,idxt] = sort(target_fitness);
        moment1_target = target_population_normalized(idxt(randi(num_front)),:);
        solution_adapted_normalized = solution_unadapted+(moment1_target-moment1_source);
    case 'M1-Tm'
        moment1_source = mean(source_population_normalized);
        moment1_target = mean(target_population_normalized);
        solution_adapted_normalized = solution_unadapted+(moment1_target-moment1_source);
    case 'M1-M'
        n = 2;
        ns = randi(n);
        nt = randi(n);
        epsilon = 1e-6;
        [~,idxs] = sort(source_fitness);
        moment1_source = mean(source_population_normalized(idxs(1:ns),:));
        [~,idxt] = sort(target_fitness);
        moment1_target = mean(target_population_normalized(idxt(1:nt),:));
        mapping_multiplication = (moment1_target+epsilon)./(moment1_source+epsilon);
        solution_adapted_normalized = solution_unadapted.*(mapping_multiplication);
    case 'M2-A'
        mu_s = mean(source_population_normalized);
        mu_t = mean(target_population_normalized);
        sigma_s = diag(diag(cov(source_population_normalized)))+eye(dim)*1e-5;
        sigma_t = diag(diag(cov(target_population_normalized)))+eye(dim)*1e-5;
        Lsi_l  = chol(inv(sigma_s));
        Lci_l = chol(inv(sigma_t));
        Am_l = inv(Lci_l')*Lsi_l;
        bm_l = mu_t'-Am_l*mu_s';
        solution_adapted_normalized = transpose(Am_l*solution_unadapted'+bm_l);
    case 'OC-L'
        [~,idxs] = sort(source_fitness);
        source_population_normalized_sort = source_population_normalized(idxs,:);
        [~,idxt] = sort(target_fitness);
        target_population_normalized_sort = target_population_normalized(idxt,:);
        M = source_population_normalized_sort\target_population_normalized_sort;
        solution_adapted_normalized = solution_unadapted*M;
    case 'OC-A'
        [~,idxs] = sort(source_fitness);
        source_population_normalized_sort = source_population_normalized(idxs,:);
        [~,idxt] = sort(target_fitness);
        target_population_normalized_sort = target_population_normalized(idxt,:);
        source_population_normalized_sort_aug = [source_population_normalized_sort,...
            ones(popsize,1)];
        M = source_population_normalized_sort_aug\target_population_normalized_sort;
        solution_unadapted_aug = [solution_unadapted 1];
        solution_adapted_normalized = solution_unadapted_aug*M;
    case 'OC-K'
        [~,idxs] = sort(source_fitness);
        source_population_normalized_sort = source_population_normalized(idxs,:);
        [~,idxt] = sort(target_fitness);
        target_population_normalized_sort = target_population_normalized(idxt,:);
        source_kernel = kernel_cal(source_population_normalized_sort,...
            source_population_normalized_sort);
        Mk = source_kernel\target_population_normalized_sort;
        transfer_kernel = kernel_cal(solution_unadapted,source_population_normalized_sort);
        solution_adapted_normalized = transfer_kernel*Mk;
    case 'OC-N'
        [~,idxs] = sort(source_fitness);
        source_population_normalized_sort = source_population_normalized(idxs,:);
        [~,idxt] = sort(target_fitness);
        target_population_normalized_sort = target_population_normalized(idxt,:);
        f_activate=@(x)1./(1+exp(-x));
        num_hiddens = dim*2;
        source_inputs = [source_population_normalized_sort ones(popsize,1)];
        target_inputs = target_population_normalized_sort;
        W_ih = rand(size(source_inputs,2),num_hiddens);
        H = f_activate(source_inputs*W_ih);
        W_ho = H\target_inputs;
        f_mapping = @(x)f_activate(x*W_ih)*W_ho;
        solution_adapted_normalized = f_mapping([solution_unadapted 1]);
    case 'ROC-L'
        num_ranklabels = 2;
        X_s = source_population_normalized;
        X_t = target_population_normalized;
        y_s = fit_relax(source_fitness,num_ranklabels);
        y_t = fit_relax(target_fitness,num_ranklabels);
        [X_sn,means,stds] = zscore(X_s);
        [X_tn,meant,stdt] = zscore(X_t);
        X_sa = X_sn';
        X_ta = X_tn';
        [ds,ns] = size(X_sa);
        [dt,nt] = size(X_ta);

        alpha = 0.1;
        d_low = 3;
        T_max = 100;
        tol = 1e-9;
        T = 1;
        floss_old = 100;
        X_total = [X_sa zeros(ds,nt);zeros(dt,ns) X_ta];
        A = [X_sa*X_sa'/ns zeros(ds,dt);zeros(dt,ds) -X_ta*X_ta'/nt];
        Ls = laplacian_matrix(y_s,y_t);
        B = X_total*(alpha*Ls)*X_total';
        [Vb,Db] = eig(B);
        [~,indb] = sort(diag(Db));
        P = real(Vb(:,indb(1:d_low)));
        while T<T_max
            floss = norm(P'*A*P,'fro')+trace(P'*B*P);
            if norm(floss-floss_old,2)<tol*floss
                break;
            end
            M = A*P*P'*A+1/2*B;
            [Vm,Dm] = eig(M);
            [~,indm] = sort(diag(Dm));
            floss_old = floss;
            P = real(Vm(:,indm(1:d_low)));
            T = T+1;
        end
        Ps1 = P(1:ds,:);
        Pt1 = P(ds+1:end,:);
        xs = (solution_unadapted-means)./stds;
        solution_adapted_normalized = transpose((Pt1*Pt1')\Pt1*(Ps1'*xs')).*stdt+meant;
    case 'SA-L'
        d = ceil(dim/2);
        coeff_source = pca(source_population_normalized);
        coeff_target = pca(target_population_normalized);
        As = coeff_source(:,1:d);
        At = coeff_target(:,1:d);
        R = diag(rand(1,d));
        M = As*As'*At*R*At';
        solution_adapted_normalized = solution_unadapted*M;
    case 'Direct'
        solution_adapted_normalized = solution_unadapted;
    case 'True'
        solution_adapted_normalized = solution_unadapted + target_opt - source_opt;
    case 'Prior'
        real_shift = target_opt - source_opt;
        [~, ada_vector, ~] = uncertainty_generate(real_shift, prior_uncertainty);
        solution_adapted_normalized = solution_unadapted + ada_vector;
end
solution_adapted_normalized(solution_adapted_normalized<0) = 0;
solution_adapted_normalized(solution_adapted_normalized>1) = 1;

parameter = solution_adapted_normalized - solution_unadapted;