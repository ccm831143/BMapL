function sigma = make_spd(sigma, epsilon)

% 1. 强制对称并减少浮点误差

sigma = (sigma + sigma') / 2;

sigma = round(sigma, 12); % 四舍五入到小数点后12位


% 2. 特征值分解与修正

[V, D] = eig(sigma);

D = diag(D);


% 确保特征值不小于 epsilon，同时避免过度平移

D = max(D, epsilon);


% 3. 重构矩阵并添加动态正则化

sigma = V * diag(D) * V';

sigma = sigma + epsilon * eye(size(sigma)) * (1 + norm(sigma, 'fro'));


% 4. 最终对称化

sigma = (sigma + sigma') / 2;

end

% 强制矩阵对称半正定
%
% 输入:
%   sigma: 待修正的矩阵
%   epsilon: 最小允许的特征值。所有小于此值的特征值都将被提升到此值。
%            通常是一个小的正数，例如 1e-6 或 1e-9。
%
% 输出:
%   sigma: 修正后的对称半正定矩阵
% function sigma = make_spd(sigma, epsilon)
%     % 1. 强制对称。
%     % 这是确保特征值分解正确（即得到实数特征值和正交特征向量）的基础。
%     % 浮点运算可能导致微小的非对称性，因此这一步至关重要，且应在分解前进行。
%     sigma = (sigma + sigma') / 2;
% 
%     % 2. 特征值分解。
%     % 对于对称矩阵，MATLAB 的 eig 函数会返回实数特征值和正交特征向量。
%     [V, D] = eig(sigma);
% 
%     % 3. 提取特征值并修正。
%     % 将对角矩阵 D 转换为向量，以便操作。
%     D = diag(D);
% 
%     % 定义一个绝对的最小特征值阈值。
%     % 这个阈值应该至少是 epsilon，并且要足够大以克服浮点误差，
%     % 确保 gmdistribution 认为矩阵是严格正定的。
%     % max(epsilon, sqrt(eps)) 是一个常见的选择，sqrt(eps) 大约是 1.49e-08。
%     % 也可以考虑一个固定的较小值，例如 1e-6。
%     min_eig_threshold = max(epsilon, sqrt(eps)); 
%     % 如果 epsilon 已经足够大 (例如 1e-4)，则 min_eig_threshold 就是 epsilon。
%     % 如果 epsilon 很小 (例如 1e-9)，则 min_eig_threshold 会是 sqrt(eps)，提供更好的鲁棒性。
% 
%     % 确保所有特征值不小于 min_eig_threshold。
%     % 任何小于此值的特征值都将被提升到此值。
%     D(D < min_eig_threshold) = min_eig_threshold;
% 
%     % 4. 使用修正后的特征值和原始特征向量重构矩阵。
%     % V * diag(D) * V' 保证了重构后的矩阵是对称的，并且具有修正后的特征值。
%     sigma = V * diag(D) * V';
% 
%     % 5. 最终对称化。
%     % 尽管理论上 V * diag(D) * V' 应该是完全对称的，但由于浮点精度问题，
%     % 在实际计算中可能会出现非常小的非对称性（例如 1e-15 级别的误差）。
%     % 这一步确保最终输出的矩阵是完全对称的，以避免后续计算中的潜在问题。
%     sigma = (sigma + sigma') / 2;
% 
%     % 6. 额外检查和修正（可选，但可以增加鲁棒性，应对极端情况）
%     % 这一步是为了确保在所有浮点误差累积后，矩阵仍然是严格正定的。
%     % 通常，如果步骤 3 的 min_eig_threshold 设置得足够好，这一步可能不是严格必需的，
%     % 但对于 gmdistribution 这种对精度要求高的函数，多一层保障是有益的。
%     current_min_eig_val = min(eig(sigma));
% 
%     % 如果最终的最小特征值仍然略小于我们设定的阈值，则添加一个小的扰动。
%     % 这里使用一个比 min_eig_threshold 更小的容差来判断。
%     if current_min_eig_val < min_eig_threshold / 2 % 容差可以根据实际情况调整
%         % 计算需要添加到对角线上的量，以确保最小特征值达到 min_eig_threshold。
%         shift_amount = min_eig_threshold - current_min_eig_val + 100 * eps; % 额外增加一个小的常数
%         sigma = sigma + max(0, shift_amount) * eye(size(sigma));
% 
%         % 再次对称化，以防添加对角项后引入微小非对称性。
%         sigma = (sigma + sigma') / 2;
%     end
% end
