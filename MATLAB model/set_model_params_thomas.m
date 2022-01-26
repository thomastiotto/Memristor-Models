clear;clc;
% load thomas_data.csv
data = importdata("Radius 10 um/-2V_3.csv");

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

%%

Vth_p = 0.5;
Vth_n = -.5;

gmin_p = 1.5e-5;
gmin_n = 4.4E-7;
bmax_p = 4.96;
bmin_p = 6.91;
bmax_n = 3.23;
bmin_n = 2.6;

gmax_p = 9e-5;
gmax_n = 1.7e-4;

% Adjust the amplitude parameters to the timescale
Ap = 90/dt_ratio
An = 10/dt_ratio

xp = .1;
xn = .242;

eta = 1;

vin = v_test;
xo = 0;

ap = 1;
an = 1; 

x = zeros(1,length(vin));
iout = zeros(1,length(vin));
t = (1:length(vin))*dt;

x(1) = xo;
for k = 2:length(vin)

    if vin(k) >= 0
        iout(k) = gmax_p*sinh(bmax_p*vin(k))*x(k-1) + gmin_p*sinh(bmin_p*vin(k))*(1-x(k-1));
    else
        iout(k) = gmax_n*sinh(bmax_n*vin(k))*x(k-1) + gmin_n*sinh(bmin_n*vin(k))*(1-x(k-1));
    end
    
    %Implement Threshold
    if vin(k) < Vth_n
        f1 = -An*(exp(-vin(k))-exp(-Vth_n));
    elseif vin(k) > Vth_p
        f1 = Ap*(exp(vin(k))-exp(Vth_p));
    else
        f1 = 0;
    end
    
    %Implement Non-Linear Boundaries
    if eta*vin(k) >= 0
        if x(k-1) >= xp
            wp = (xp - x(k-1)) / (1 - xp) + 1;
            f2 = exp(-ap*(x(k-1)-xp))*wp;
        else
            f2 = 1;
        end
    else
        if x(k-1) <= xn
            wn = x(k-1) / xn;
            f2 = exp(an*(x(k-1)-xn))*wn;   
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

