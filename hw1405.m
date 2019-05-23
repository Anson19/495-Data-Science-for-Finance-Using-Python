%% MGTF 405 HW 1
%% 
% Qingyue Shen from ???

%% Part 1

% 1.  file reading
unrate = xlsread('time_series_data_2019.xls',1);
cpiaucsl = xlsread('time_series_data_2019.xls',2);
goldp = xlsread('time_series_data_2019.xls',3);
sp500 = xlsread('time_series_data_2019.xls',4);
sp500(:,1) = [];

timeVec1 = (1948:1/12:2018+11/12)';
timeVec2 = (1947:1/12:2018+10/12)';
timeVec3 = (1940:1/12:2018+11/12)';
timeVec4 = (1871:1/12:2018+11/12)';

figure
plot(timeVec1,unrate)
xlim([timeVec1(1) timeVec1(end)])
xlabel('Time')
ylabel('Unemployment rate')

figure
plot(timeVec2,cpiaucsl)
xlim([timeVec2(1) timeVec2(end)])
xlabel('Time')
ylabel('cpiaucsl')

figure
plot(timeVec3,goldp)
xlim([timeVec3(1) timeVec3(end)])
xlabel('Time')
ylabel('gold price')

figure
plot(timeVec4,sp500)
xlim([timeVec4(1) timeVec4(end)])
xlabel('Time')
ylabel('sp500')

% figure 2,3,4 are all time-varying, use figure 5,6,7 to replace 2,3,4
% respectively
% doing log return for 2,3,4
lgun = logrt(unrate);
lgcpi = logrt(cpiaucsl);
lggol = logrt(goldp);
lgsp  = logrt(sp500);
timeVec1(1) = [];
timeVec2(1) = [];
timeVec3(1) = [];
timeVec4(1) = [];

figure
plot(timeVec1,lgun)
xlim([timeVec1(1) timeVec1(end)])
xlabel('Time')
ylabel('log-unrate')

figure
plot(timeVec2,lgcpi)
xlim([timeVec2(1) timeVec2(end)])
xlabel('Time')
ylabel('log-cpiaucsl')

figure
plot(timeVec3,lggol)
xlim([timeVec3(1) timeVec3(end)])
xlabel('Time')
ylabel('log-gold price')

figure
plot(timeVec4,lgsp)
xlim([timeVec4(1) timeVec4(end)])
xlabel('Time')
ylabel('log-sp500')

%% 2.
% using columns unrate, lgcpi, lggol, lgsp
% as figure 8,9,10,11
figure
autocorr(lgun,10)
xlim([0 10])
title('Unemployment rate serial correlation tests')
disp('10 lags')
[h1,pValue1] = lbqtest(unrate,'lags',10)

figure
autocorr(lgcpi,10)
xlim([0 10])
title('CPI serial correlation tests')
disp('10 lags')
[h2,pValue2] = lbqtest(lgcpi,'lags',10)

figure
autocorr(lggol,10)
xlim([0 10])
title('Gold price serial correlation tests')
disp('10 lags')
[h3,pValue3] = lbqtest(lggol,'lags',10)

figure
autocorr(lgsp,10)
xlim([0 10])
title('Sp500 index serial correlation tests')
disp('10 lags')
[h4,pValue4] = lbqtest(lgsp,'lags',10)

% unrate is very persistent, cpi is somewhat persistent, while the other
% two are not persistent at all
% p-val all small enough, statistically significant

%% 3.
% I've no idea how this works, but got the answer anyway, function in
% function list. RUNNING SLOW!!!
best1 = arimatest(lgun);
% for unrate, best arima is [3,3]
best2 = arimatest(lgcpi);
% for cpi, best arima is [4,3]
best3 = arimatest(lggol);
% for gold price, best arima is [4,2]
best4 = arimatest(lgsp);
% for sp500, best arima is [2,4]

%% 4.
[fit1,pval1] = gof(lgun,best1);
[fit2,pval2] = gof(lgcpi,best2);
[fit3,pval3] = gof(lggol,best3);
[fit4,pval4] = gof(lgsp,best4);

% p-values = 0.4196, 0.1177, 0.5136, 0.1246

% unrate comparison figure as figure 12
figure
plot(timeVec1,[fit1 lgun])
legend('Fitted values','Data')
title('unrate gof')
xlim([timeVec1(1) timeVec1(end)])
hold on
plot(timeVec1,lgun)
legend('Original values','Data')
hold off

% cpi comparison figure (13)
figure
plot(timeVec2,[fit2 lgcpi])
legend('Fitted values','Data')
title('cpi gof')
xlim([timeVec2(1) timeVec2(end)])
hold on
plot(timeVec2,lgcpi)
legend('Original values','Data')
hold off

% gold price comparison figure (14)
figure
plot(timeVec3,[fit3 lggol])
legend('Fitted values','Data')
title('gold price gof')
xlim([timeVec3(1) timeVec3(end)])
hold on
plot(timeVec3,lggol)
legend('Original values','Data')
hold off

% sp500 comparison figure (15)

figure
plot(timeVec4,[fit4 lgsp])
legend('Fitted values','Data')
title('sp500 gof')
xlim([timeVec4(1) timeVec4(end)])
hold on
plot(timeVec4,lgsp)
legend('Original values','Data')
hold off


%% q2

%% 1.
data = xlsread('Keeling_CO2data_2019.xlsx');
co2 = data(:,5);
co2(1) = [];
co2(1) = [];
timeVec = (1958:1/12:2018+11/12)';
timeVec(1) = [];
timeVec(1) = [];

figure
plot(timeVec,co2)
title('co2 emission time series original')
xlabel('time')
ylabel('co2 emission')

% upward trending, time varying, covariance stable, no jump observed

%% 2.
%398.79
co2short = co2(1:562);
timeVecshort = timeVec(1:562);

% linear regression
[alpha,beta] = reg(timeVecshort,co2short);
% y = -2620+1.495x

figure
plot(timeVec,co2)
title('co2 emission time series original')
xlabel('time')
ylabel('co2 emission')
hold on
y = [];
for i = 1:562
    y(i) = timeVecshort(i)*beta + alpha;
end
plot(timeVecshort,y)
legend('data','linear fit')
hold off
% not that bad, but not that good, becaue of an upward trending

%% 3. 
% trying quadratic function to include trend
p = polyfit(timeVecshort,co2short,2);
y1 = polyval(p,timeVecshort);
figure
plot(timeVecshort,[y1 co2short])
legend('quadratic fit','Data')
xlabel('time')
ylabel('co2 level')
title('co2 quadratic fit')

%% 4. 
% performing arima like q1, is that correct?
[fit,pval] = gof(co2short,[2,2]);

figure
plot(timeVecshort,[fit co2short])
title('co2 arima')
legend('Fitted values','Data')
xlabel('time')
ylabel('co2 level')
xlim([timeVecshort(1) timeVecshort(end)])
legend('Original values','Data')

% way better, showing all fluctuations, trends. the two curve almost
% overlap.

%% 5.
% still, using the arima(1,1)
mdl = arima('Constant',0,'D',1,'Seasonality',12,...
    'MALags',1,'SMALags',12);
EstMdl = estimate(mdl,co2);
[yf,yfMSE] = forecast(EstMdl,168,'Y0',co2short);
%upper = yf + 1.96*sqrt(yfMSE);
%lower = yf - 1.96*sqrt(yfMSE);
T = 562;
timeVecelse = timeVec(563:730);

figure
plot(timeVecelse,yf)
hold on
plot(timeVec,co2)
legend('forecast','original')
%h1 = plot(T+1:T+168,yf,'r','LineWidth',2);
%h2 = plot(T+1:T+168,upper,'k--','LineWidth',1.5);
%plot(T+1:T+168,lower,'k--','LineWidth',1.5)
%xlim([0,T+168])
title('Forecast co2')
hold off

%% 6.
% suppose human popolation as new variable, dataset from 1992.1 to 2018.10
%pop = xlsread('PSD_Monthly_Population_Counts_by_Facility.csv','P2:P220');
%pop = pop/12;
%co2pop = co2(407:728);
%timeVecHa = (1996:1/12:2014+2/12)';

gdp = xlsread('popdata');
timeVecHa = (1992:1/12:2018+9/12)'; 
co2gdp = co2(407:728);
gdp = gdp/30; % scaled by 30 for visual reason on plot

figure
plot(timeVecHa,[gdp co2gdp])
title('scaled GDP and co2 emission')
legend('scaled GDP and co2','co2')
% showing a similar trend, thus a good factor to add into model
corrco = corr(gdp,co2gdp);
% 0.9717. Good correlation between two factors.


%% 7.
% still, using the arima(1,1)
[yf2,yf2MSE] = forecast(EstMdl,120,'Y0',co2);
upper = yf2 + 1.96*sqrt(yf2MSE);
lower = yf2 - 1.96*sqrt(yf2MSE);
timeVecfuture = (2019:1/12:2028+11/12)';

figure
plot(timeVecfuture,yf2)
hold on
plot(timeVec,co2)
legend('forecast','original','upper bond','lower bond')
plot(timeVecfuture,upper,'k--','LineWidth',1.5);
plot(timeVecfuture,lower,'k--','LineWidth',1.5)
%xlim([0,T+121])
title('Forecast co2')
hold off

%%
% function list

function rt = logrt(data)
for i = 2:length(data)
    rt(i,1) = log(data(i)) - log(data(i-1));
end
rt(1) = '';
end

function [bestAICModelLags] = arimatest(data)
maxAR = 4; 
maxMA = 4;
ModelCriteria1 = zeros((maxAR+1)*(maxMA+1)-1,2);
ind = 1;
lagCombo1 = ModelCriteria1;
for ii = 0:maxAR
    for jj = 0:maxMA
        if ii ~= 0 || jj ~= 0
            model = arima(ii,0,jj);
            [~,~,logL] = estimate(model,data);
            [ModelCriteria1(ind,1),ModelCriteria1(ind,2)] = aicbic(logL,ii+jj,length(data)-max(ii,jj));
            lagCombo1(ind,:) = [ii jj];
            ind = ind + 1;
        end
    end
end
[~,minIndices1] = min(ModelCriteria1);
bestAICModelLags = lagCombo1(minIndices1(1),:);
end

function [datFit,pValue] = gof(data,best)
model = arima(best(1,1),0,best(1,2));
modelEstimate = estimate(model,data);

resid = infer(modelEstimate,data);
[h,pValue] = lbqtest(resid,'lags',10)
datFit = data - resid;
end

function [c,phi] = reg(x,y)
x = [ones(length(x),1) x];
b = regress(y,x);
c = b(1);
phi = b(2);
end
