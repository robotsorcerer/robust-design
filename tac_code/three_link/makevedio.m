figure(1)
hAxes(1) = gca;
hAnimatedLine(1) = animatedline( hAxes(1),'Color','b','LineWidth',1.3 )
hAnimatedLine(2) = animatedline( hAxes(1),'Color','r','LineWidth',1.3 )
hAnimatedLine(3) = animatedline( hAxes(1),'Color','g','LineWidth',1.3 )
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',23)
ylabel({'$\mathbf{\theta}$ (deg)'},'Interpreter','latex','FontSize',23)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', '$\mathbf{\theta_3}$', 'Interpreter','latex', 'FontSize',20)
title('\bf Joint Angles','Interpreter','latex','FontSize',23)
set(gca,'FontSize',23)
grid on
set(gca,'XLim',[0,1])
set(gca,'YLim',[-100,100])
hold on

figure(2)
hAxes(2) = gca;
hAnimatedLine(4) = animatedline( hAxes(2),'Color','b','LineWidth',1.3 )
hAnimatedLine(5) = animatedline( hAxes(2),'Color','r','LineWidth',1.3 )
hAnimatedLine(6) = animatedline( hAxes(2),'Color','g','LineWidth',1.3 )
xlabel({'Time (s)'},'Interpreter','latex','FontSize',23)
ylabel({'$\dot{\theta}$ (deg/s)'},'Interpreter','latex','FontSize',23)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$', '$\mathbf{\dot{\theta}_3}$','Interpreter','latex', 'FontSize',23)
title('\bf Joint Velocities','Interpreter','latex','FontSize',23)
set(gca,'FontSize',23)
set(gca,'XLim',[0,1])
set(gca,'YLim',[-1000,1000])
grid on
hold on


for i = 1:1000
    
    addpoints(hAnimatedLine(1),i*dt-dt,X_lqr(i,1))
    addpoints(hAnimatedLine(2),i*dt-dt,X_lqr(i,2))
    addpoints(hAnimatedLine(3),i*dt-dt,X_lqr(i,3))
    
    addpoints(hAnimatedLine(4),i*dt-dt,X_lqr(i,4));
    addpoints(hAnimatedLine(5),i*dt-dt,X_lqr(i,5));
    addpoints(hAnimatedLine(6),i*dt-dt,X_lqr(i,6));
    
    drawnow   
end