static double dj_o4[5][5][5] = {
    {{1.0,0.0, -0.5, -0.0,0.375},
    {0.0, -0.70710678, -0.0,0.4330127,0.},
    {0.0,0.0,0.61237244,0.0, -0.39528471},
    {0.0,0.0,0.0, -0.55901699, -0.},
    {0.0,0.0,0.0,0.0,0.52291252}
    },
    {{0.0,0.70710678,0.0, -0.4330127, -0.},
    {0.0,0.5, -0.5, -0.125,0.375},
    {0.0,0.0, -0.5,0.39528471,0.1767767},
    {0.0,0.0,0.0,0.48412292, -0.33071891},
    {0.0,0.0,0.0,0.0, -0.46770717}
    },
    {{0.0,0.0,0.61237244,0.0, -0.39528471},
    {0.0,0.0,0.5, -0.39528471, -0.1767767},
    {0.0,0.0,0.25, -0.5,0.25},
    {0.0,0.0,0.0, -0.30618622,0.46770717},
    {0.0,0.0,0.0,0.0,0.33071891}
    },
    {{0.0,0.0,0.0,0.55901699,0.},
    {0.0,0.0,0.0,0.48412292, -0.33071891},
    {0.0,0.0,0.0,0.30618622, -0.46770717},
    {0.0,0.0,0.0,0.125, -0.375},
    {0.0,0.0,0.0,0.0, -0.1767767}
    },
    {{0.0,0.0,0.0,0.0,0.52291252},
    {0.0,0.0,0.0,0.0,0.46770717},
    {0.0,0.0,0.0,0.0,0.33071891},
    {0.0,0.0,0.0,0.0,0.1767767},
    {0.0,0.0,0.0,0.0,0.0625}}};


void map_dipy_to_pysh_o4(double* dipy_v, double* sh) {
    int clmax = 5;

    sh[(0 * clmax + 0) * clmax + 0] = dipy_v[0];
    sh[(1 * clmax + 2) * clmax + 2] = dipy_v[1];
    sh[(1 * clmax + 2) * clmax + 1] = dipy_v[2];
    sh[(0 * clmax + 2) * clmax + 0] = dipy_v[3];
    sh[(0 * clmax + 2) * clmax + 1] = dipy_v[4];
    sh[(0 * clmax + 2) * clmax + 2] = dipy_v[5];
    sh[(1 * clmax + 4) * clmax + 4] = dipy_v[6];
    sh[(1 * clmax + 4) * clmax + 3] = dipy_v[7];
    sh[(1 * clmax + 4) * clmax + 2] = dipy_v[8];
    sh[(1 * clmax + 4) * clmax + 1] = dipy_v[9];
    sh[(0 * clmax + 4) * clmax + 0] = dipy_v[10];
    sh[(0 * clmax + 4) * clmax + 1] = dipy_v[11];
    sh[(0 * clmax + 4) * clmax + 2] = dipy_v[12];
    sh[(0 * clmax + 4) * clmax + 3] = dipy_v[13];
    sh[(0 * clmax + 4) * clmax + 4] = dipy_v[14];
}

void map_pysh_to_dipy_o4(double* sh, double* dipy_v) {
    int clmax = 5;

    dipy_v[0] =  sh[(0 * clmax + 0) * clmax + 0];
    dipy_v[1] =  sh[(1 * clmax + 2) * clmax + 2];
    dipy_v[2] =  sh[(1 * clmax + 2) * clmax + 1];
    dipy_v[3] =  sh[(0 * clmax + 2) * clmax + 0];
    dipy_v[4] =  sh[(0 * clmax + 2) * clmax + 1];
    dipy_v[5] =  sh[(0 * clmax + 2) * clmax + 2];
    dipy_v[6] =  sh[(1 * clmax + 4) * clmax + 4];
    dipy_v[7] =  sh[(1 * clmax + 4) * clmax + 3];
    dipy_v[8] =  sh[(1 * clmax + 4) * clmax + 2];
    dipy_v[9] =  sh[(1 * clmax + 4) * clmax + 1];
    dipy_v[10] = sh[(0 * clmax + 4) * clmax + 0];
    dipy_v[11] = sh[(0 * clmax + 4) * clmax + 1];
    dipy_v[12] = sh[(0 * clmax + 4) * clmax + 2];
    dipy_v[13] = sh[(0 * clmax + 4) * clmax + 3];
    dipy_v[14] = sh[(0 * clmax + 4) * clmax + 4];
}

double Fk = dawson(sqrt(kappa));
void sh_watson_coeffs(double kappa, double* dipy_v, int lmax) {
    dipy_v[0] = 0.28209479177387814;// = 1 / (4*pi) * 2 * sqrt(M_PI)
    dipy_v[3] = 1 / (4*M_PI) * (3 / (sqrt(kappa)*Fk) - 3/kappa -2)*M_PI * sqrt(5 / (4*M_PI));
    dipy_v[10] = 1 / (4*M_PI) * (5*sqrt(kappa)*(2*kappa-21)/Fk + 12*pow(kappa,2) + 60*kappa + 105)*1/(8*pow(kappa,2))*M_PI * sqrt(9 / (4*M_PI));
    if (lmax > 4) {
        dipy_v[21] = 1 / (4*M_PI) * (21*sqrt(kappa)*(4*(kappa-5)*kappa+165) / Fk - 5*(8*pow(kappa,3)+84*pow(kappa,2)+378*kappa+693))*1/(32*pow(kappa,3))*M_PI * sqrt(13 / (4*M_PI));
    }
    if (lmax > 6) {
        dipy_v[36] = 1 / (4*M_PI) * M_PI / (512.0 * pow(kappa,4)) * ((3*sqrt(kappa)*(2*kappa*(2*kappa*(62*kappa-1925)+15015)-225225.0)) / Fk + 35*(8*kappa*(kappa*(2*kappa*(kappa + 18)+297)+1287)+19305)) * sqrt(19 / (4*M_PI));
    }
}


void SHrtoc(double* ccilm, double* rcilm, int lmax) {
    int max = lmax+1;

    for (int l = -1; l < lmax; l++) {
        ccilm[(0 * max + (l+1)) * max + 0] = std::sqrt(4.0*M_PI) * rcilm[(0 * max + (l+1)) * max + 0];
        ccilm[(1 * max + (l+1)) * max + 0] = 0.0;

        for (int m = 0; m < l+1; m++) {
            ccilm[(0 * max + (l+1)) * max + (m+1)] = std::sqrt(2.0*M_PI) * rcilm[(0 * max + (l+1)) * max + (m+1)] * pow(-1,m+1);
            ccilm[(1 * max + (l+1)) * max + (m+1)] = -std::sqrt(2.0*M_PI) * rcilm[(1 * max + (l+1)) * max + (m+1)] * pow(-1,m+1);
        }
    }
}

void SHctor(double* ccilm, double* rcilm, int lmax) {
    int max = lmax+1;

    for (int l = -1; l < lmax; l++) {
        rcilm[(0 * max + (l+1)) * max + 0] = ccilm[(0 * max + (l+1)) * max + 0] / std::sqrt(4.0*M_PI);
        rcilm[(1 * max + (l+1)) * max + 0] = 0.0;

        for (int m = 0; m < l+1; m++) {
            rcilm[(0 * max + (l+1)) * max + (m+1)] = ccilm[(0 * max + (l+1)) * max + (m+1)] / std::sqrt(2.0*M_PI) * pow(-1,m+1);
            rcilm[(1 * max + (l+1)) * max + (m+1)] = -ccilm[(1 * max + (l+1)) * max + (m+1)] / std::sqrt(2.0*M_PI) * pow(-1,m+1);
        }
    }
}

void SHCilmToCindex(double* cilm, double* cindex, int lmax) {
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;
    for (int l = -1; l < lmax; l++) {
        for (int m = -1; m < l+1; m++) {
            int index = ((l+1)*(l+2))/2+m+1;
            cindex[0 * cimax + index] = cilm[(0 * clmax + (l+1)) * clmax + (m+1)];
            cindex[1 * cimax + index] = cilm[(1 * clmax + (l+1)) * clmax + (m+1)];
        }
    }
}

void SHCindexToCilm(double* cindex, double* cilm, int lmax) {
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;

    for (int l = -1; l < lmax; l++) {
        for (int m = -1; m < l+1; m++) {
            int index = ((l+1)*(l+2))/2 +m+1;
            cilm[(0 * clmax + (l+1)) * clmax + (m+1)] = cindex[0 * cimax + index];
            cilm[(1 * clmax + (l+1)) * clmax + (m+1)] = cindex[1 * cimax + index];
        }
    }
}

void SHRotateCoef(double* x, double* cof, double* rcof, double* dj, int lmax) {
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;

    double sum[2], temp[2][lmax+1], temp2[2][lmax+1], cgam[lmax+1],
            sgam[lmax+1], calf[lmax+1], salf[lmax+1], cbet[lmax+1], sbet[lmax+1];

    double pi2 = M_PI_2;

    double alpha = x[0];
    double beta = x[1];
    double gamma = x[2];

    alpha = alpha - pi2;
    gamma = gamma + pi2;
    beta = -beta;

    int ind = 0;

    // all degrees
    for (int lp1 = 1; lp1 <= lmax+1; lp1++) {
        int l = lp1-1;
        cbet[lp1-1] = cos(l*beta);
        sbet[lp1-1] = sin(l*beta);
        cgam[lp1-1] = cos(l*gamma);
        sgam[lp1-1] = sin(l*gamma);
        calf[lp1-1] = cos(l*alpha);
        salf[lp1-1] = sin(l*alpha);


        // rotation around alpha angle
        for (int mp1 = 1; mp1 <= lp1; mp1++) {
            int indx = ind+mp1;
            temp[0][mp1-1] = cof[0 * cimax + indx-1] * calf[mp1-1] - cof[1 * cimax + indx-1] * salf[mp1-1];
            temp[1][mp1-1] = cof[1 * cimax + indx-1] * calf[mp1-1] + cof[0 * cimax + indx-1] * salf[mp1-1];
        }

        // first step of euler decomposition followed by rotation around beta angle
        for (int jp1 = 1; jp1 <= lp1; jp1++) {
            sum[0] = dj[((jp1-1) * clmax + 0) * clmax + (lp1-1)] * temp[0][0];
            sum[1] = 0.0;
            int isgn = 1 - 2 * ((lp1-jp1) % 2);

            for (int mp1 = 2; mp1 <= lp1; mp1++) {
                isgn = -isgn;
                int ii = (3-isgn) / 2;
                sum[ii-1] = sum[ii-1] + 2.0 * dj[((jp1-1) * clmax + (mp1-1)) * clmax + (lp1-1)] * temp[ii-1][mp1-1];
            }

            temp2[0][jp1-1] = sum[0] * cbet[jp1-1] - sum[1] * sbet[jp1-1];
            temp2[1][jp1-1] = sum[1] * cbet[jp1-1] + sum[0] * sbet[jp1-1];
        }

        // second step of euler decomposition followed by rotation around gamma angle
        for (int jp1 = 1; jp1 <= lp1; jp1++) {
            sum[0] = dj[(0 * clmax + (jp1-1)) * clmax + (lp1-1)] * temp2[0][0];
            sum[1] = 0.0;
            int isgn = 1 - 2 * ((lp1-jp1) % 2);

            for (int mp1 = 2; mp1 <= lp1; mp1++) {
                isgn = -isgn;
                int ii = (3-isgn) / 2;
                sum[ii-1] = sum[ii-1] + 2.0 * dj[((mp1-1) * clmax + (jp1-1)) * clmax + (lp1-1)] * temp2[ii-1][mp1-1];
            }

            int indx = ind + jp1;
            rcof[0 * cimax + indx-1] = sum[0] * cgam[jp1-1] - sum[1] * sgam[jp1-1];
            rcof[1 * cimax + indx-1] = sum[1] * cgam[jp1-1] + sum[0] * sgam[jp1-1];
        }

        ind = ind + lp1;
    }
}


void SHRotateRealCoef(double* cilmrot, double* cilm, int lmax, double* x) {
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;

    double ccilmd[2][clmax][clmax];
    double cindex[2][cimax];

    // all steps of real sh rotation
    SHrtoc(&ccilmd[0][0][0], cilm, lmax);
    SHCilmToCindex(&ccilmd[0][0][0], &cindex[0][0], lmax);
    SHRotateCoef(x, &cindex[0][0], &cindex[0][0], dj_o4, lmax);
    SHCindexToCilm(&cindex[0][0], &ccilmd[0][0][0], lmax);
    SHctor(&ccilmd[0][0][0], cilmrot, lmax);
}