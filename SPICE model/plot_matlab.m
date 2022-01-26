raw_data = LTspice2Matlab('./willian/exp-1_0.raw');

disp( sprintf('\n\nThis file contains %.0f variables:\n', raw_data.num_variables) );
disp( sprintf('NAME         TYPE\n-------------------------') );
disp( [char(raw_data.variable_name_list), char(zeros(raw_data.num_variables,5)), ...
       char(raw_data.variable_type_list)] );

% This example plots all variables in the data structure.
figure;
plot(raw_data.time_vect, raw_data.variable_mat);
title(sprintf( 'File:  %s', raw_data.title));
legend(raw_data.variable_name_list);
ylabel('Voltage (V) or Current (A)');
xlabel('Time (sec)');

% This example plots the first variable in the data structure.
figure;
variable_to_plot = 1;   
plot( raw_data.time_vect, raw_data.variable_mat(variable_to_plot,:), 'k' );
title( sprintf('Waveform %s', raw_data.variable_name_list{variable_to_plot}) );
ylabel( raw_data.variable_type_list{variable_to_plot} );
xlabel( 'Time (sec)' );