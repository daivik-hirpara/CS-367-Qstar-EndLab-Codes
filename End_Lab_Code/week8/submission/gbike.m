clear all;
close all;

Lamda = [3 4];
lamda = [3 2];
r = 10;
t = 2;
parking_penalty = 4;

policy = zeros(21, 21);
gam = 0.9;

policystable = false;
count = 0;
while ~policystable
    V = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam, parking_penalty);
    [policy, policystable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, parking_penalty);
    count = count + 1;
end

figure(1);
subplot(2, 1, 1); contour(policy, [-5:5]);
subplot(2, 1, 2); surf(V);

function [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, parking_penalty)
[m, n] = size(policy);

nn = 0:n - 1;
P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

policy_stable = true;
old_policy = policy;

for i = 1:m
    for j = 1:n
        s1 = i - 1; s2 = j - 1;
        amin = -min(min(s2, m - 1 - s1), 5); 
        amax = min(min(s1, n - 1 - s2), 5);
        v_ = -inf;
        for a = amin:amax
            R = -max(0, abs(a) - 1) * t; 
            if s1 - a > 10
                R = R - parking_penalty;
            end
            if s2 + a > 10
                R = R - parking_penalty;
            end
            Vs_ = 0;
            s1_ = s1 - a; s2_ = s2 + a;
            for n1 = 0:12
                for n2 = 0:14
                    s1__ = s1_ - min(n1, s1_); s2__ = s2_ - min(n2, s2_);
                    for n3 = 0:12
                        for n4 = 0:9
                            s1___ = s1__ + min(n3, 20 - s1__);
                            s2___ = s2__ + min(n4, 20 - s2__);
                            Vs_ = Vs_ + P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1);
                            R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                        end
                    end
                end
            end
            if (R + gam * Vs_ > v_)
                v_ = R + gam * Vs_;
                policy(i, j) = a;
            end
        end
    end
end

if sum(sum(abs(old_policy - policy))) ~= 0
    policy_stable = false;
end
end

function [V] = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam, parking_penalty)
[m, n] = size(policy);

nn = 0:n - 1;
P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

V = zeros(m, n);
delta = 10;
theta = 0.1;

while delta > theta
    v = V;
    for i = 1:m
        for j = 1:n
            s1 = i - 1; s2 = j - 1;
            Vs_ = 0; a = policy(i, j);
            R = -max(0, abs(a) - 1) * t; 
            if s1 - a > 10
                R = R - parking_penalty;
            end
            if s2 + a > 10
                R = R - parking_penalty;
            end
            s1_ = s1 - a; s2_ = s2 + a;
            for n1 = 0:12
                for n2 = 0:14
                    s1__ = s1_ - min(n1, s1_); s2__ = s2_ - min(n2, s2_);
                    for n3 = 0:12
                        for n4 = 0:9
                            s1___ = s1__ + min(n3, 20 - s1__);
                            s2___ = s2__ + min(n4, 20 - s2__);
                            R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                            Vs_ = Vs_ + P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1);
                        end
                    end
                end
            end
            V(i, j) = R + (gam * Vs_);
        end
    end
    delta = max(max(abs(v - V)));
end
end
