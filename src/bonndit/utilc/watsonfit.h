#ifndef WATSONFIT_H
#define WATSONFIT_H

void minimize_watson_mult_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread);
void minimize_watson_mult_o6(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread);
void minimize_watson_mult_o8(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread);
void SHRotateRealCoef(double*, double*, int, double*, double*);
void SHRotateRealCoefFast(double*, int, double*, int,  int, double*, double*);
void map_dipy_to_pysh_o4(double*, double*);
void map_pysh_to_dipy_o4(double*, double*);
void sh_watson_coeffs(double, double*, int);

void map_pysh_to_dipy_o4_scaled(double, double* , double* , int );

#endif