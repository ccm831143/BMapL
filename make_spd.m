function sigma = make_spd(sigma, epsilon)


sigma = (sigma + sigma') / 2;

sigma = round(sigma, 12);


[V, D] = eig(sigma);

D = diag(D);

D = max(D, epsilon);


sigma = V * diag(D) * V';

sigma = sigma + epsilon * eye(size(sigma)) * (1 + norm(sigma, 'fro'));


sigma = (sigma + sigma') / 2;

end

