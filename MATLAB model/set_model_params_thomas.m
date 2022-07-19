clear;clc;
% load thomas_data.csv
data = importdata("Radius 10 um/-2V_0.csv");

%% Device Data

%Assume 100pts/sec data collection
dt_old = 1/10000
%Use real timestep
dt = mean(diff(data.T))
dt_ratio=dt/dt_old

% v = -flipud(data(:,3));
% i = -flipud(data(:,4));

v = data.V;
i = data.I;

v_test = v;
i_test = i;
g_test = i./v;
t_test = (1:length(v))*dt;

% tiledlayout(3,1)
% nexttile
% plot(t_test,v_test)
% nexttile
% plot(t_test, i_test)
% nexttile
% plot(v_test, i_test)

%% OLD
% Vp = 0.5;
% Vn = -.5;
%
% Ap=90;
% An=10;
%
% gmin_p = 1.5e-5;
% gmin_n = 4.4E-7;
% bmax_p = 4.96;
% bmin_p = 6.91;
% bmax_n = 3.23;
% bmin_n = 2.6;
%
% gmax_p = 9e-5;
% gmax_n = 1.7e-4;
%
% xp = .1;
% xn = .242;
%
% alphap = 1;
% alphan = 1;
%
% eta = 1;
%
% xo=0;

%% NEW
gmax_p=0.0004338454236;
bmax_p=4.988561168;
gmax_n=8.44e-6;
bmax_n=6.272960721;
gmin_p=0.03135053798;
bmin_p=0.002125127287;
gmin_n=1.45e-05;
bmin_n=3.295533935;
Ap=0.071;
An=0.02662694665;
Vp=0;
Vn=0;
xp=0.11;
xn=0.1433673316;
alphap=9.2;
alphan=0.7013461469;
xo=0;
eta=1;

% Adjust the amplitude parameters to the timescale
% Ap = Ap/dt_ratio
% An = An/dt_ratio



vin = v_test;

x = zeros(1,length(vin));
iout = zeros(1,length(vin));
t = (1:length(vin))*dt;

x(1) = xo;
for k = 2:length(vin)

    if vin(k) >= 0
        iout(k) = gmax_p*sinh(bmax_p*vin(k))*x(k-1) + gmin_p*(1-exp(-bmin_p*vin(k)))*(1-x(k-1));
    else
        iout(k) = gmax_n*(1-exp(-bmax_n*vin(k)))*x(k-1) + gmin_n*sinh(bmin_n*vin(k))*(1-x(k-1));
    end

    %Implement Threshold
    if vin(k) < Vn
        f1 = -An*(exp(-vin(k))-exp(-Vn));
    elseif vin(k) > Vp
        f1 = Ap*(exp(vin(k))-exp(Vp));
    else
        f1 = 0;
    end

    %Implement Non-Linear Boundaries
    if eta*vin(k) >= 0
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

    log_f2(k) = f2;
    
%     FE
    x(k) = x(k-1) + eta*f1*f2*dt;
    f1*f2*dt;
    %This is not necessary if 'dt' is sufficiently small
    if x(k) < 0
        x(k) = 0;
    end
    if x(k) > 1
        x(k) = 1;
    end
     
    
end
%%
tiledlayout(3,1)
nexttile
plot(t,x,"g-",'LineWidth',2)
set(gca,'YColor','g');
ylabel('State Variable','Color','g')
xlabel('Time (s)')
axis tight
nexttile
plot(v_test,i_test*1000,'r-',vin,1000*iout,'k-','LineWidth',2)
ylabel('Current (mA)')
xlabel('Voltage (V)')
axis tight
nexttile
yyaxis left
plot(t,i_test*1000,'r-',t,iout*1000,'k-','LineWidth',2)
ylabel('Current (mA)','Color','k')
set(gca,'YColor','k');
axis tight
yyaxis right
plot(t,vin,'b-','LineWidth',2)
ylabel('Voltage (V)','Color','b')
xlabel('Time (s)')
axis tight
set(gca,'YColor','b');

