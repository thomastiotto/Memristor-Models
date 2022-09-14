%% LAST UPDATED 12/08/2022 - Thomas Tiotto (t.f.tiotto@rug.nl)
% Interpolated time and voltage vectors using dt
% Added ability to switch between new (Alina) and old (Thomas) models
% Added Nengo timestep adjustment for new model

%% Import data
clear;clc;

data = importdata("Radius 10 um/-2V_0.csv");

v = data.V;
i = data.I;
t = data.T;

dt_data = mean(diff(data.T))

%% OLD PARAMETERS (Thomas)
model='new'

if strcmp(model,'old')
    % # internal state parameters
    Ap=90;
    An=10;
    Vp = 0.5;
    Vn = -.5;
    xp = .1;
    xn = .242;
    alphap = 1;
    alphan = 1;
    % electron transfer parameters
    gmax_p = 9e-5;
    bmax_p = 4.96;
    gmax_n = 1.7e-4;
    bmax_n = 3.23;
    gmin_p = 1.5e-5;
    bmin_p = 6.91;
    gmin_n = 4.4E-7;
    bmin_n = 2.6;
    % simulation parameters
    eta = 1;
    x0=0;
    
    %Assume 100pts/sec data collection
    dt_thomas = 1/10000
    dt_ratio=dt_data/dt_thomas
    
    % Adjust the amplitude parameters to the timescale
    Ap = Ap/dt_ratio
    An = An/dt_ratio
    
    % Simulate using data-inferred timestep
    dt=dt_data;
elseif strcmp(model,'new')
    % NEW PARAMETERS (Alina)
    % # internal state parameters
    Ap=0.071;
    An=0.02662694665;
    Vp=0;
    Vn=0;
    xp=0.11;
    xn=0.1433673316;
    alphap=9.2;
    alphan=0.7013461469;
    eta=1;
    % electron transfer parameters
    gmax_p=0.0004338454236;
    bmax_p=4.988561168;
    gmax_n=8.44e-6;
    bmax_n=6.272960721;
    gmin_p=0.03135053798;
    bmin_p=0.002125127287;
    gmin_n=1.45e-05;
    bmin_n=3.295533935;
    % simulation parameters
    x0=0;

    % Nengo timestep
    dt_nengo=1e-3
    dt_ratio=dt_nengo/dt_data
    
    % Adjust the amplitude parameters to the timescale
%     Ap = Ap/dt_ratio
%     An = An/dt_ratio
    
    % Simulate using Nengo timestep
    dt=dt_nengo;
end

%% Expand values using dt
tint=0:dt:t(end);
vint=interp1(t,v,tint);
iint=interp1(t,i,tint,'spline');

t=tint;
v=vint;
i=iint;

%% Simulate
x = zeros(1,length(v));
iout = zeros(1,length(v));

x(1) = x0;
for k = 2:length(v)
    if strcmp(model,'old')
        if v(k) >= 0
            iout(k) = gmax_p*sinh(bmax_p*v(k))*x(k-1) + gmin_p*sinh(bmin_p*v(k))*(1-x(k-1));
        else
            iout(k) = gmax_n*sinh(bmax_n*v(k))*x(k-1) + gmin_n*sinh(bmin_n*v(k))*(1-x(k-1));
        end
    end

    if strcmp(model,'new')
        if v(k) >= 0
            iout(k) = gmax_p*sinh(bmax_p*v(k))*x(k-1) + gmin_p*(1-exp(-bmin_p*v(k)))*(1-x(k-1));
        else
            iout(k) = gmax_n*(1-exp(-bmax_n*v(k)))*x(k-1) + gmin_n*sinh(bmin_n*v(k))*(1-x(k-1));
        end
    end

    %Implement Threshold
    if v(k) < Vn
        f1 = -An*(exp(-v(k))-exp(-Vn));
    elseif v(k) > Vp
        f1 = Ap*(exp(v(k))-exp(Vp));
    else
        f1 = 0;
    end

    %Implement Non-Linear Boundaries
    if eta*v(k) >= 0
        if x(k-1) >= xp
            wp = (xp - x(k-1)) / (1 - xp) + 1;
            f2 = exp(-alphap*(x(k-1)-xp))*wp;
        else
            f2 = 1;
        end
    else
        if x(k-1) <= xn
            wn = x(k-1) / xn;
            f2 = exp(alphan*(x(k-1)-xn))*wn;
        else
            f2 = 1;
        end
    end  
%     FE
    x(k) = x(k-1) + eta*f1*f2*dt;
    %This is not necessary if 'dt' is sufficiently small
    if x(k) < 0
        x(k) = 0;
    end
    if x(k) > 1
        x(k) = 1;
    end    
end

% Plot
tiledlayout(3,1)
nexttile
plot(t,x,"g-",'LineWidth',2)
set(gca,'YColor','g');
ylabel('State Variable','Color','g')
xlabel('Time (s)')
axis tight
nexttile
plot(v,i*1000,'r-',v,iout*1000,'k-','LineWidth',2)
ylabel('Current (mA)')
xlabel('Voltage (V)')
axis tight
nexttile
yyaxis left
plot(t,i*1000,'r-',t,iout*1000,'k-','LineWidth',2)
ylabel('Current (mA)','Color','k')
set(gca,'YColor','k');
axis tight
yyaxis right
plot(t,v,'b-','LineWidth',2)
ylabel('Voltage (V)','Color','b')
xlabel('Time (s)')
axis tight
set(gca,'YColor','b');

