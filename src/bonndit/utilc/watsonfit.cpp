#include "watsonfit.h"
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <omp.h>
#include "cerf.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <stdexcept>

static double* signal;
static double* est_signal;
static double* dipy_v;
static double* pysh_v;
static double* rot_pysh_v;
static double* angles_v;

static double* parameters_v;
static double* loss_v;
static int lmax;
static int num_of_dir;

// rotation matrices for sh rotation
static double dj_o8[9][9][9] = {
{{1.0, 0.0, -0.5, -0.0, 0.37499999999999994, 0.0, -0.31249999999999994, -0.0, 0.27343750000000006},
{0.0, -0.7071067811865475, -0.0, 0.4330127018922193, 0.0, -0.3423265984407288, -0.0, 0.2923169833417142, 0.0},
{0.0, 0.0, 0.6123724356957945, 0.0, -0.39528470752104744, -0.0, 0.32021721143623744, 0.0, -0.2773162398327945},
{0.0, 0.0, 0.0, -0.5590169943749475, -0.0, 0.36975498644372606, 0.0, -0.30378472023786846, -0.0},
{0.0, 0.0, 0.0, 0.0, 0.5229125165837972, 0.0, -0.350780380010057, -0.0, 0.2908517260779107},
{0.0, 0.0, 0.0, 0.0, 0.0, -0.49607837082461076, -0.0, 0.3358466446906981, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47495887979908324, 0.0, -0.3236299246438747},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4576818286211503, -0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44314852502786806},
},
{{0.0, 0.7071067811865475, 0.0, -0.4330127018922193, -0.0, 0.3423265984407288, 0.0, -0.2923169833417142, -0.0},
{0.0, 0.5, -0.5, -0.12499999999999999, 0.375, 0.06250000000000003, -0.3125, -0.0390625, 0.2734375},
{0.0, 0.0, -0.5, 0.39528470752104744, 0.17677669529663692, -0.33071891388307384, -0.09882117688026182, 0.2870495792324037, 0.06536406457297465},
{0.0, 0.0, 0.0, 0.4841229182759271, -0.3307189138830738, -0.2025231468252457, 0.29646353064078557, 0.12178482240718669, -0.26551008543697546},
{0.0, 0.0, 0.0, 0.0, -0.46770717334674267, 0.28641098093474, 0.21650635094610962, -0.2692763740564701, -0.13710881855300192},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.45285552331841994, -0.2538762001448738, -0.22439697838039174, 0.2471764378055684},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4397264774834465, 0.22884091431057516, 0.22884091431057518},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42812214871313303, -0.20890221808001466},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4178044361600293},
},
{{0.0, 0.0, 0.6123724356957945, 0.0, -0.39528470752104744, -0.0, 0.32021721143623744, 0.0, -0.2773162398327945},
{0.0, 0.0, 0.5, -0.39528470752104744, -0.17677669529663692, 0.33071891388307384, 0.09882117688026182, -0.2870495792324037, -0.06536406457297465},
{0.0, 0.0, 0.25, -0.5, 0.24999999999999997, 0.25, -0.26562500000000006, -0.15625, 0.25},
{0.0, 0.0, 0.0, -0.30618621784789724, 0.46770717334674267, -0.1530931089239486, -0.28125, 0.20992232566475633, 0.19040715010865528},
{0.0, 0.0, 0.0, 0.0, 0.33071891388307384, -0.4330127018922193, 0.08558164961018223, 0.29315098498896436, -0.16387638252658618},
{0.0, 0.0, 0.0, 0.0, 0.0, -0.3423265984407288, 0.4014135180832853, -0.036643873123620566, -0.2954323500185787},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3476343040826092, -0.3736956482219187, -0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3495602706437046, 0.3495602706437046},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3495602706437046},
},
{{0.0, 0.0, 0.0, 0.5590169943749475, 0.0, -0.36975498644372606, -0.0, 0.30378472023786846, 0.0},
{0.0, 0.0, 0.0, 0.4841229182759271, -0.3307189138830738, -0.2025231468252457, 0.29646353064078557, 0.12178482240718669, -0.26551008543697546},
{0.0, 0.0, 0.0, 0.30618621784789724, -0.46770717334674267, 0.1530931089239486, 0.28125, -0.20992232566475633, -0.19040715010865528},
{0.0, 0.0, 0.0, 0.125, -0.375, 0.40624999999999994, -0.031249999999999965, -0.3046875, 0.13281250000000006},
{0.0, 0.0, 0.0, 0.0, -0.1767766952966369, 0.397747564417433, -0.3423265984407288, -0.05182226234930306, 0.3025768239224545},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.2096313728906053, -0.4014135180832853, 0.28502244292116724, 0.10909562534194485},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.23175620272173947, 0.396364090436432, -0.2356734863905993},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24717643780556836, -0.387251054106054},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.258167369404036},
},
{{0.0, 0.0, 0.0, 0.0, 0.5229125165837972, 0.0, -0.350780380010057, -0.0, 0.2908517260779107},
{0.0, 0.0, 0.0, 0.0, 0.46770717334674267, -0.28641098093474, -0.21650635094610962, 0.2692763740564701, 0.13710881855300192},
{0.0, 0.0, 0.0, 0.0, 0.33071891388307384, -0.4330127018922193, 0.08558164961018223, 0.29315098498896436, -0.16387638252658618},
{0.0, 0.0, 0.0, 0.0, 0.1767766952966369, -0.397747564417433, 0.3423265984407288, 0.05182226234930306, -0.3025768239224545},
{0.0, 0.0, 0.0, 0.0, 0.0625, -0.25, 0.40625000000000006, -0.25000000000000006, -0.14062500000000003},
{0.0, 0.0, 0.0, 0.0, 0.0, -0.09882117688026186, 0.29315098498896436, -0.390625, 0.16901021603737448},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1269381000724369, -0.31868871959954903, 0.3651037951733726},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.14905300022139775, 0.3332926407453366},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1666463203726683},
},
{{0.0, 0.0, 0.0, 0.0, 0.0, 0.49607837082461076, 0.0, -0.3358466446906981, -0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.45285552331841994, -0.2538762001448738, -0.22439697838039174, 0.2471764378055684},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.3423265984407288, -0.4014135180832853, 0.036643873123620566, 0.2954323500185787},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.2096313728906053, -0.4014135180832853, 0.28502244292116724, 0.10909562534194485},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.09882117688026186, -0.29315098498896436, 0.390625, -0.16901021603737448},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.03125, -0.15625, 0.3359375, -0.3515625},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05412658773652741, 0.19918044974971816, -0.35441550694417984},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07452650011069888, -0.23109686652732875},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0924387466109315},
},
{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47495887979908324, 0.0, -0.3236299246438747},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4397264774834465, -0.22884091431057516, -0.22884091431057518},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3476343040826092, -0.3736956482219187, -0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23175620272173947, -0.396364090436432, 0.2356734863905993},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1269381000724369, -0.31868871959954903, 0.3651037951733726},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05412658773652741, -0.19918044974971816, 0.35441550694417984},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015625, -0.09375, 0.24999999999999997},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.029231698334171417, 0.1283724744152733},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0427908248050911},
},
{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4576818286211503, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42812214871313303, -0.20890221808001466},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3495602706437046, -0.3495602706437046},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24717643780556836, -0.387251054106054},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14905300022139775, -0.3332926407453366},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07452650011069888, -0.23109686652732875},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.029231698334171417, -0.1283724744152733},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0078125, -0.0546875},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.015625},
},
{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44314852502786806},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4178044361600293},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3495602706437046},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.258167369404036},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1666463203726683},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0924387466109315},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0427908248050911},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015625},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00390625},
},
};

static double dj_o6[7][7][7] = {
    {{1.0 ,0.0 ,-0.5 ,-0.0 ,0.37499999999999994 ,0.0 ,-0.31249999999999994},
    {0.0 ,-0.7071067811865475 ,-0.0 ,0.4330127018922193 ,0.0 ,-0.3423265984407288 ,-0.0},
    {0.0 ,0.0 ,0.6123724356957945 ,0.0 ,-0.39528470752104744 ,-0.0 ,0.32021721143623744},
    {0.0 ,0.0 ,0.0 ,-0.5590169943749475 ,-0.0 ,0.36975498644372606 ,0.0},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.5229125165837972 ,0.0 ,-0.350780380010057},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.49607837082461076 ,-0.0},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.47495887979908324},
    },
    {{0.0 ,0.7071067811865475 ,0.0 ,-0.4330127018922193 ,-0.0 ,0.3423265984407288 ,0.0},
    {0.0 ,0.5 ,-0.5 ,-0.12499999999999999 ,0.375 ,0.06250000000000003 ,-0.3125},
    {0.0 ,0.0 ,-0.5 ,0.39528470752104744 ,0.17677669529663692 ,-0.33071891388307384 ,-0.09882117688026182},
    {0.0 ,0.0 ,0.0 ,0.4841229182759271 ,-0.3307189138830738 ,-0.2025231468252457 ,0.29646353064078557},
    {0.0 ,0.0 ,0.0 ,0.0 ,-0.46770717334674267 ,0.28641098093474 ,0.21650635094610962},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.45285552331841994 ,-0.2538762001448738},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.4397264774834465},
    },
    {{0.0 ,0.0 ,0.6123724356957945 ,0.0 ,-0.39528470752104744 ,-0.0 ,0.32021721143623744},
    {0.0 ,0.0 ,0.5 ,-0.39528470752104744 ,-0.17677669529663692 ,0.33071891388307384 ,0.09882117688026182},
    {0.0 ,0.0 ,0.25 ,-0.5 ,0.24999999999999997 ,0.25 ,-0.26562500000000006},
    {0.0 ,0.0 ,0.0 ,-0.30618621784789724 ,0.46770717334674267 ,-0.1530931089239486 ,-0.28125},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.33071891388307384 ,-0.4330127018922193 ,0.08558164961018223},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.3423265984407288 ,0.4014135180832853},
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.3476343040826092},
    },
    {{0.0 ,0.0 ,0.0 ,0.5590169943749475 ,0.0 ,-0.36975498644372606 ,-0.0},
    {0.0 ,0.0 ,0.0 ,0.4841229182759271 ,-0.3307189138830738 ,-0.2025231468252457 ,0.29646353064078557},
    {0.0 ,0.0 ,0.0 ,0.30618621784789724 ,-0.46770717334674267 ,0.1530931089239486 ,0.28125},
    {0.0 ,0.0 ,0.0 ,0.125 ,-0.375 ,0.40624999999999994 ,-0.031249999999999965},
    {0.0 ,0.0 ,0.0 ,0.0 ,-0.1767766952966369 ,0.397747564417433 ,-0.3423265984407288 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.2096313728906053 ,-0.4014135180832853 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.23175620272173947 },
    },
    {{0.0 ,0.0 ,0.0 ,0.0 ,0.5229125165837972 ,0.0 ,-0.350780380010057 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.46770717334674267 ,-0.28641098093474 ,-0.21650635094610962 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.33071891388307384 ,-0.4330127018922193 ,0.08558164961018223 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.1767766952966369 ,-0.397747564417433 ,0.3423265984407288 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0625 ,-0.25 ,0.40625000000000006 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.09882117688026186 ,0.29315098498896436 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.1269381000724369 },
    },
    {{0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.49607837082461076 ,0.0 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.45285552331841994 ,-0.2538762001448738 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.3423265984407288 ,-0.4014135180832853 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.2096313728906053 ,-0.4014135180832853 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.09882117688026186 ,-0.29315098498896436 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.03125 ,-0.15625 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.05412658773652741 },
    },
    {{0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.47495887979908324 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.4397264774834465 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.3476343040826092 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.23175620272173947 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.1269381000724369 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.05412658773652741 },
    {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.015625 }}};

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

// diagonal entries of rank1 rotation harmonics
static double rank1_rh_o4[3] = {2.51327412, 1.43615664, 0.31914592};
static double rank1_rh_o6[4] = {1.7951958 , 1.1967972 , 0.43519898, 0.06695369};
static double rank1_rh_o8[8] = {1.3962634 , 1.01546429, 0.46867583, 0.12498022, 0.01470356};

/// helper class for showing the progressbar
/// adapted on the basis of https://stackoverflow.com/a/44555438
class Timer{
 private:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::duration<double, std::ratio<1> > second;

  std::chrono::time_point<clock> start_time;
  double accumulated_time;
  bool running;

 public:
  Timer(){
    accumulated_time = 0;
    running          = false;
  }

  void start(){
    if(running)
      throw std::runtime_error("Timer was already started!");
    running=true;
    start_time = clock::now();
  }

  double stop(){
    if(!running)
      throw std::runtime_error("Timer was already stopped!");

    accumulated_time += lap();
    running           = false;

    return accumulated_time;
  }

  double accumulated(){
    if(running)
      throw std::runtime_error("Timer is still running!");
    return accumulated_time;
  }

  double lap(){
    if(!running)
      throw std::runtime_error("Timer was not started!");
    return std::chrono::duration_cast<second> (clock::now() - start_time).count();
  }

  void reset(){
    accumulated_time = 0;
    running          = false;
  }
};

/// helper class for showing the progressbar
/// adapted on the basis of https://stackoverflow.com/a/44555438
class ProgressBar{
 private:
  uint32_t total_work;
  uint32_t next_update;
  uint32_t call_diff;
  uint32_t work_done;
  uint16_t old_percent;
  Timer    timer;

  void clearConsoleLine() const {
    std::cerr<<"\r\033[2K"<<std::flush;
  }

 public:
  void start(uint32_t total_work){
    timer = Timer();
    timer.start();
    this->total_work = total_work;
    next_update      = 0;
    call_diff        = (total_work > 800) ? total_work/800 : 5;
    old_percent      = 0;
    work_done        = 0;
    clearConsoleLine();
  }

  void update(uint32_t work_done0){
    #ifdef NOPROGRESS
      return;
    #endif

    if(omp_get_thread_num()!=0)
      return;
    work_done = work_done0;
    if(work_done<next_update)
      return;
    next_update += call_diff;
    uint16_t percent = (uint8_t)(work_done*omp_get_num_threads()*100/total_work);
    if(percent>100)
      percent=100;

    if(percent==old_percent)
      return;

    old_percent=percent;
    int remaining_time = int(timer.lap()/percent*(100-percent));
    std::cerr<<"\r\033[2K|"
             <<std::string(percent/2, '=')<<std::string(50-percent/2, ' ')
             <<"| "
             <<percent<<"% ~ "
             <<std::fixed<<std::setprecision(1)<<((remaining_time > 120) ? remaining_time / 60 : remaining_time)
             <<((remaining_time > 120) ? "m" : "s")<<" remaining - "
             <<omp_get_num_threads()<< " threads   "<<std::flush;
  }

  ProgressBar& operator++(){
    if(omp_get_thread_num()!=0)
      return *this;

    work_done++;
    update(work_done);
    return *this;
  }

  double stop(){
    clearConsoleLine();

    timer.stop();
    return timer.accumulated();
  }

  double time_it_took(){
    return timer.accumulated();
  }

  uint32_t cellsProcessed() const {
    return work_done;
  }
};

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

void SHRotateRealCoef(double* cilmrot, double* cilm, int lmax, double* x, double* dj) {
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;

    double ccilmd[2][clmax][clmax];
    double cindex[2][cimax];

    // all steps of real sh rotation
    SHrtoc(&ccilmd[0][0][0], cilm, lmax);
    SHCilmToCindex(&ccilmd[0][0][0], &cindex[0][0], lmax);
    SHRotateCoef(x, &cindex[0][0], &cindex[0][0], dj, lmax);
    SHCindexToCilm(&cindex[0][0], &ccilmd[0][0][0], lmax);
    SHctor(&ccilmd[0][0][0], cilmrot, lmax);
}

void map_pysh_to_dipy_o8(double* sh, double* dipy_v) {
    int clmax = 9;

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
    dipy_v[15] = sh[(1 * clmax + 6) * clmax + 6];
    dipy_v[16] = sh[(1 * clmax + 6) * clmax + 5];
    dipy_v[17] = sh[(1 * clmax + 6) * clmax + 4];
    dipy_v[18] = sh[(1 * clmax + 6) * clmax + 3];
    dipy_v[19] = sh[(1 * clmax + 6) * clmax + 2];
    dipy_v[20] = sh[(1 * clmax + 6) * clmax + 1];
    dipy_v[21] = sh[(0 * clmax + 6) * clmax + 0];
    dipy_v[22] = sh[(0 * clmax + 6) * clmax + 1];
    dipy_v[23] = sh[(0 * clmax + 6) * clmax + 2];
    dipy_v[24] = sh[(0 * clmax + 6) * clmax + 3];
    dipy_v[25] = sh[(0 * clmax + 6) * clmax + 4];
    dipy_v[26] = sh[(0 * clmax + 6) * clmax + 5];
    dipy_v[27] = sh[(0 * clmax + 6) * clmax + 6];
    dipy_v[28] = sh[(1 * clmax + 8) * clmax + 8];
    dipy_v[29] = sh[(1 * clmax + 8) * clmax + 7];
    dipy_v[30] = sh[(1 * clmax + 8) * clmax + 6];
    dipy_v[31] = sh[(1 * clmax + 8) * clmax + 5];
    dipy_v[32] = sh[(1 * clmax + 8) * clmax + 4];
    dipy_v[33] = sh[(1 * clmax + 8) * clmax + 3];
    dipy_v[34] = sh[(1 * clmax + 8) * clmax + 2];
    dipy_v[35] = sh[(1 * clmax + 8) * clmax + 1];
    dipy_v[36] = sh[(0 * clmax + 8) * clmax + 0];
    dipy_v[37] = sh[(0 * clmax + 8) * clmax + 1];
    dipy_v[38] = sh[(0 * clmax + 8) * clmax + 2];
    dipy_v[39] = sh[(0 * clmax + 8) * clmax + 3];
    dipy_v[40] = sh[(0 * clmax + 8) * clmax + 4];
    dipy_v[41] = sh[(0 * clmax + 8) * clmax + 5];
    dipy_v[42] = sh[(0 * clmax + 8) * clmax + 6];
    dipy_v[43] = sh[(0 * clmax + 8) * clmax + 7];
    dipy_v[44] = sh[(0 * clmax + 8) * clmax + 8];
}
 
void map_pysh_to_dipy_o6(double* sh, double* dipy_v) {
    int clmax = 7;

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
    dipy_v[15] = sh[(1 * clmax + 6) * clmax + 6];
    dipy_v[16] = sh[(1 * clmax + 6) * clmax + 5];
    dipy_v[17] = sh[(1 * clmax + 6) * clmax + 4];
    dipy_v[18] = sh[(1 * clmax + 6) * clmax + 3];
    dipy_v[19] = sh[(1 * clmax + 6) * clmax + 2];
    dipy_v[20] = sh[(1 * clmax + 6) * clmax + 1];
    dipy_v[21] = sh[(0 * clmax + 6) * clmax + 0];
    dipy_v[22] = sh[(0 * clmax + 6) * clmax + 1];
    dipy_v[23] = sh[(0 * clmax + 6) * clmax + 2];
    dipy_v[24] = sh[(0 * clmax + 6) * clmax + 3];
    dipy_v[25] = sh[(0 * clmax + 6) * clmax + 4];
    dipy_v[26] = sh[(0 * clmax + 6) * clmax + 5];
    dipy_v[27] = sh[(0 * clmax + 6) * clmax + 6];
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

void map_dipy_to_pysh_o8(double* dipy_v, double* sh) {
    int clmax = 9;

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
    sh[(1 * clmax + 6) * clmax + 6] = dipy_v[15];
    sh[(1 * clmax + 6) * clmax + 5] = dipy_v[16];
    sh[(1 * clmax + 6) * clmax + 4] = dipy_v[17];
    sh[(1 * clmax + 6) * clmax + 3] = dipy_v[18];
    sh[(1 * clmax + 6) * clmax + 2] = dipy_v[19];
    sh[(1 * clmax + 6) * clmax + 1] = dipy_v[20];
    sh[(0 * clmax + 6) * clmax + 0] = dipy_v[21];
    sh[(0 * clmax + 6) * clmax + 1] = dipy_v[22];
    sh[(0 * clmax + 6) * clmax + 2] = dipy_v[23];
    sh[(0 * clmax + 6) * clmax + 3] = dipy_v[24];
    sh[(0 * clmax + 6) * clmax + 4] = dipy_v[25];
    sh[(0 * clmax + 6) * clmax + 5] = dipy_v[26];
    sh[(0 * clmax + 6) * clmax + 6] = dipy_v[27];
    sh[(1 * clmax + 8) * clmax + 8] = dipy_v[28];
    sh[(1 * clmax + 8) * clmax + 7] = dipy_v[29];
    sh[(1 * clmax + 8) * clmax + 6] = dipy_v[30];
    sh[(1 * clmax + 8) * clmax + 5] = dipy_v[31];
    sh[(1 * clmax + 8) * clmax + 4] = dipy_v[32];
    sh[(1 * clmax + 8) * clmax + 3] = dipy_v[33];
    sh[(1 * clmax + 8) * clmax + 2] = dipy_v[34];
    sh[(1 * clmax + 8) * clmax + 1] = dipy_v[35];
    sh[(0 * clmax + 8) * clmax + 0] = dipy_v[36];
    sh[(0 * clmax + 8) * clmax + 1] = dipy_v[37];
    sh[(0 * clmax + 8) * clmax + 2] = dipy_v[38];
    sh[(0 * clmax + 8) * clmax + 3] = dipy_v[39];
    sh[(0 * clmax + 8) * clmax + 4] = dipy_v[40];
    sh[(0 * clmax + 8) * clmax + 5] = dipy_v[41];
    sh[(0 * clmax + 8) * clmax + 6] = dipy_v[42];
    sh[(0 * clmax + 8) * clmax + 7] = dipy_v[43];
    sh[(0 * clmax + 8) * clmax + 8] = dipy_v[44];
}

void map_dipy_to_pysh_o6(double* dipy_v, double* sh) {
    int clmax = 7;

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
    sh[(1 * clmax + 6) * clmax + 6] = dipy_v[15];
    sh[(1 * clmax + 6) * clmax + 5] = dipy_v[16];
    sh[(1 * clmax + 6) * clmax + 4] = dipy_v[17];
    sh[(1 * clmax + 6) * clmax + 3] = dipy_v[18];
    sh[(1 * clmax + 6) * clmax + 2] = dipy_v[19];
    sh[(1 * clmax + 6) * clmax + 1] = dipy_v[20];
    sh[(0 * clmax + 6) * clmax + 0] = dipy_v[21];
    sh[(0 * clmax + 6) * clmax + 1] = dipy_v[22];
    sh[(0 * clmax + 6) * clmax + 2] = dipy_v[23];
    sh[(0 * clmax + 6) * clmax + 3] = dipy_v[24];
    sh[(0 * clmax + 6) * clmax + 4] = dipy_v[25];
    sh[(0 * clmax + 6) * clmax + 5] = dipy_v[26];
    sh[(0 * clmax + 6) * clmax + 6] = dipy_v[27];
}

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

void sh_watson_coeffs(double kappa, double* dipy_v, int lmax) {
    double Fk = dawson(sqrt(kappa));
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

double watson_minimizer(const double* x, double* signal, double* est_signal, double* dipy_v, double* pysh_v, double* rot_pysh_v, double* angles, double* dj, int num_of_dir, int lmax, int no_spread) {
    double weight, kappa, diff, peak_value, loss = 0;
    int clmax = lmax+1;
    int cimax = (lmax*(lmax+1))/2+lmax+1;

    // reset arrays
    for (int j = 0; j < cimax; j++) {
        dipy_v[j] = 0.0;
    }

    for (int j = 0; j < cimax; j++) {
        est_signal[j] = 0.0;
    }

    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < clmax; k++) {
            for (int l = 0; l < clmax; l++) {
                pysh_v[(j * clmax + k) * clmax + l] = 0.0;
            }
        }
    }

    // compute loss for all distributions
    for (int i = 0; i < num_of_dir; i++) {
        weight = abs(x[i*4]);
        kappa = exp(x[i*4+1]);
        angles[1] = x[i*4+2];
        angles[2] = x[i*4+3];

        // reset coefficients for next distribution
        for (int j = 0; j < cimax; j++) {
            dipy_v[j] = 0.0;
        }
        
        // set watson coefficients if we want to fit watson, else we only fit the lowrank tensors
        if (no_spread == 0) {
            sh_watson_coeffs(kappa, dipy_v, lmax);
        } else {
            if (lmax >= 4) {
                dipy_v[0] = dipy_v[3] = dipy_v[10] = 1;
                if (lmax >= 6) {
                    dipy_v[21] = 1;
                    if (lmax >= 8) {
                        dipy_v[36] = 1;
                    }
                }
            }
        }

        // convolution of rank1 tensor in RH with watson in SH
        if (lmax == 4) {
            dipy_v[0] *= rank1_rh_o4[0];
            dipy_v[3] *= rank1_rh_o4[1];
            dipy_v[10] *= rank1_rh_o4[2];
        } else if (lmax == 6) {
            dipy_v[0] *= rank1_rh_o6[0];
            dipy_v[3] *= rank1_rh_o6[1];
            dipy_v[10] *= rank1_rh_o6[2];
            dipy_v[21] *= rank1_rh_o6[3];
        } else if (lmax == 8) {
            dipy_v[0] *= rank1_rh_o8[0];
            dipy_v[3] *= rank1_rh_o8[1];
            dipy_v[10] *= rank1_rh_o8[2];
            dipy_v[21] *= rank1_rh_o8[3];
            dipy_v[36] *= rank1_rh_o8[4];
        }

        // rotate the distribution
        if (lmax == 4) {
            map_dipy_to_pysh_o4(dipy_v, pysh_v);
            SHRotateRealCoef(rot_pysh_v, pysh_v, lmax, angles, &dj_o4[0][0][0]);
            map_pysh_to_dipy_o4(rot_pysh_v, dipy_v);
        } else if (lmax == 6) {
            map_dipy_to_pysh_o6(dipy_v, pysh_v);
            SHRotateRealCoef(rot_pysh_v, pysh_v, lmax, angles, &dj_o6[0][0][0]);
            map_pysh_to_dipy_o6(rot_pysh_v, dipy_v);
        } else if (lmax == 8) {
            map_dipy_to_pysh_o8(dipy_v, pysh_v);
            SHRotateRealCoef(rot_pysh_v, pysh_v, lmax, angles, &dj_o8[0][0][0]);
            map_pysh_to_dipy_o8(rot_pysh_v, dipy_v);
        }

        // add to combined signal
        for (int j = 0; j < cimax; j++) {
            est_signal[j] += dipy_v[j] * weight;
        }
    }

    // compute loss
    for (int i = 0; i < cimax; i++) {
        diff = signal[i] - est_signal[i];
        loss += diff * diff;
    }

    return loss;
}

struct WatsonSHApprox {
    WatsonSHApprox(double* local_signal, double* local_est_signal, double* local_dipy_v, double* local_pysh_v, double* local_rot_pysh_v, double* local_angles_v, double* local_dj, int local_num_of_dir, int local_lmax, int local_no_spread) : 
                 local_signal(local_signal), local_est_signal(local_est_signal), local_dipy_v(local_dipy_v), local_pysh_v(local_pysh_v), local_rot_pysh_v(local_rot_pysh_v), local_angles_v(local_angles_v), local_dj(local_dj), local_num_of_dir(local_num_of_dir), local_lmax(local_lmax), local_no_spread(local_no_spread) {}
    
    bool operator()(const double* parameters, double* cost) const {
        cost[0] = watson_minimizer(parameters, local_signal, local_est_signal, local_dipy_v, local_pysh_v, local_rot_pysh_v, local_angles_v, local_dj, local_num_of_dir, local_lmax, local_no_spread);
        return true;
    }
    private:
        double* local_signal;
        double* local_est_signal;
        double* local_dipy_v;
        double* local_pysh_v;
        double* local_rot_pysh_v;
        double* local_angles_v;
        double* local_dj;
        int local_num_of_dir;
        int local_lmax;
        int local_no_spread;
};

// watson minimization at sh order 4
void minimize_watson_mult_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) {
    int local_lmax = 4;
    ProgressBar pg;
    
    if (amount != 1) {
        // set amount of threads
        int nProcessors = omp_get_max_threads();
        omp_set_num_threads(nProcessors);
        
        // init progressbar
        pg.start(amount);

        // parallel fitting
        #pragma omp parallel for schedule(static)
        for (int i=0; i<amount; i++) {
            pg.update(i);

            ceres::GradientProblemSolver::Options options;
            options.minimizer_progress_to_stdout = false;
            options.logging_type = ceres::SILENT;

            ceres::GradientProblemSolver::Summary summary;

            constexpr int kNumParameters2 = 8;
            constexpr int kNumParameters3 = 12;

            if (num_of_dir_p == 2) {
                // initialize ceres optimizer for two distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                        new WatsonSHApprox( &signal_p[i * 15], 
                                            &est_signal_p[i * 15], 
                                            &dipy_v_p[i * 15], 
                                            &pysh_v_p[i * 2 * 5 * 5], 
                                            &rot_pysh_v_p[i * 2 * 5 * 5], 
                                            &angles_v_p[i * 3], 
                                            &dj_o4[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );

                // run optimization
                ceres::Solve(options, problem, &parameters[i * 8], &summary);
            } else {
                // initialize ceres optimizer for three distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                        new WatsonSHApprox( &signal_p[i * 15], 
                                            &est_signal_p[i * 15], 
                                            &dipy_v_p[i * 15], 
                                            &pysh_v_p[i * 2 * 5 * 5], 
                                            &rot_pysh_v_p[i * 2 * 5 * 5], 
                                            &angles_v_p[i * 3], 
                                            &dj_o4[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );

                // run optimization
                ceres::Solve(options, problem, &parameters[i * 12], &summary);
            }

            loss_p[i] = summary.final_cost;
        }
    } else {
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary;

        constexpr int kNumParameters2 = 8;
        constexpr int kNumParameters3 = 12;
        
        if (num_of_dir_p == 2) {
            // initialize ceres optimizer for two distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o4[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );

            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        } else {
            // initialize ceres optimizer for three distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o4[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );

            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        }

        loss_p[0] = summary.final_cost;
    }
}

// watson minimization at sh order 6
void minimize_watson_mult_o6(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) {
    int local_lmax = 6;
    ProgressBar pg;
    
    if (amount != 1) {
        // set amount of threads
        int nProcessors = omp_get_max_threads();
        omp_set_num_threads(nProcessors);

        pg.start(amount);

        #pragma omp parallel for schedule(static)
        for (int i=0; i<amount; i++) {
            pg.update(i);

            ceres::GradientProblemSolver::Options options;
            options.minimizer_progress_to_stdout = false;
            options.logging_type = ceres::SILENT;

            ceres::GradientProblemSolver::Summary summary;

            constexpr int kNumParameters2 = 8;
            constexpr int kNumParameters3 = 12;

            if (num_of_dir_p == 2) {
                // initialize ceres optimizer for two distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                        new WatsonSHApprox( &signal_p[i * 28], 
                                            &est_signal_p[i * 28], 
                                            &dipy_v_p[i * 28], 
                                            &pysh_v_p[i * 2 * 7 * 7], 
                                            &rot_pysh_v_p[i * 2 * 7 * 7], 
                                            &angles_v_p[i * 3], 
                                            &dj_o6[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );

                // run optimization
                ceres::Solve(options, problem, &parameters[i * 8], &summary);
            } else {
                // initialize ceres optimizer for three distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                        new WatsonSHApprox( &signal_p[i * 28], 
                                            &est_signal_p[i * 28], 
                                            &dipy_v_p[i * 28], 
                                            &pysh_v_p[i * 2 * 7 * 7], 
                                            &rot_pysh_v_p[i * 2 * 7 * 7], 
                                            &angles_v_p[i * 3], 
                                            &dj_o6[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );

                // run optimization
                ceres::Solve(options, problem, &parameters[i * 12], &summary);
            }

            loss_p[i] = summary.final_cost;
        }
    } else {
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary;

        constexpr int kNumParameters2 = 8;
        constexpr int kNumParameters3 = 12;
        
        if (num_of_dir_p == 2) {
            // initialize ceres optimizer for two distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o6[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );

            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        } else {
            // initialize ceres optimizer for three distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o6[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );
            
            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        }

        loss_p[0] = summary.final_cost;
    }
}

// watson minimization at sh order 8
void minimize_watson_mult_o8(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) {
    int local_lmax = 8;
    ProgressBar pg;
    
    if (amount != 1) {
        // set amount of threads
        int nProcessors = omp_get_max_threads();
        omp_set_num_threads(nProcessors);

        pg.start(amount);

        #pragma omp parallel for schedule(static)
        for (int i=0; i<amount; i++) {
            pg.update(i);

            ceres::GradientProblemSolver::Options options;
            options.minimizer_progress_to_stdout = false;
            options.logging_type = ceres::SILENT;

            ceres::GradientProblemSolver::Summary summary;

            constexpr int kNumParameters2 = 8;
            constexpr int kNumParameters3 = 12;

            if (num_of_dir_p == 2) {
                // initialize ceres optimizer for two distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, 8>(
                        new WatsonSHApprox( &signal_p[i * 45], 
                                            &est_signal_p[i * 45], 
                                            &dipy_v_p[i * 45], 
                                            &pysh_v_p[i * 2 * 9 * 9], 
                                            &rot_pysh_v_p[i * 2 * 9 * 9], 
                                            &angles_v_p[i * 3], 
                                            &dj_o8[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );
                
                // run optimization
                ceres::Solve(options, problem, &parameters[i * 8], &summary);
            } else {
                // initialize ceres optimizer for three distributions
                ceres::GradientProblem problem(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, 12>(
                        new WatsonSHApprox( &signal_p[i * 45], 
                                            &est_signal_p[i * 45], 
                                            &dipy_v_p[i * 45], 
                                            &pysh_v_p[i * 2 * 9 * 9], 
                                            &rot_pysh_v_p[i * 2 * 9 * 9], 
                                            &angles_v_p[i * 3], 
                                            &dj_o8[0][0][0], 
                                            num_of_dir_p,
                                            local_lmax,
                                            no_spread)
                        )
                    );

                // run optimization
                ceres::Solve(options, problem, &parameters[i * 12], &summary);
            }

            loss_p[i] = summary.final_cost;
        }
    } else {
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary;

        constexpr int kNumParameters2 = 8;
        constexpr int kNumParameters3 = 12;
        
        if (num_of_dir_p == 2) {
            // initialize ceres optimizer for two distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o8[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );
            
            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        } else {
            // initialize ceres optimizer for three distributions
            ceres::GradientProblem problem(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                    new WatsonSHApprox( &signal_p[0], 
                                        &est_signal_p[0], 
                                        &dipy_v_p[0], 
                                        &pysh_v_p[0], 
                                        &rot_pysh_v_p[0], 
                                        &angles_v_p[0], 
                                        &dj_o8[0][0][0], 
                                        num_of_dir_p,
                                        local_lmax,
                                        no_spread)
                    )
                );

            // run optimization
            ceres::Solve(options, problem, &parameters[0], &summary);
        }

        loss_p[0] = summary.final_cost;
    }
}

// iterative optimization at sh order 4 - experimental
void minimize_watson_mult_iterative_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* dj_p, double* loss_p, int amount, int lmax_p, int num_of_dir_p) {
    
    // set amount of threads
    int nProcessors = omp_get_max_threads();
    //std::cout<<nProcessors<<std::endl;
    omp_set_num_threads(nProcessors);

    #pragma omp parallel for 
    for (int i=0; i<amount; i++) {
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = false;//true;
        options.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary;

        constexpr int kNumParameters = 4;

        ceres::GradientProblem problem(
            new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters>(
                new WatsonSHApprox( &signal_p[i * 15], 
                                    &est_signal_p[i * 15], 
                                    &dipy_v_p[i * 15], 
                                    &pysh_v_p[i * 2 * 5 * 5], 
                                    &rot_pysh_v_p[i * 2 * 5 * 5], 
                                    &angles_v_p[i * 3], 
                                    &dj_o4[0][0][0], 
                                    1,
                                    4,
                                    0)
                )
            );
        
        ceres::Solve(options, problem, &parameters[i * 12], &summary);

        ceres::GradientProblemSolver::Options options2;
        options2.minimizer_progress_to_stdout = false;//true;
        options2.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary2;

        constexpr int kNumParameters2 = 12;

        ceres::GradientProblem problem2(
            new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                new WatsonSHApprox( &signal_p[i * 15], 
                                    &est_signal_p[i * 15], 
                                    &dipy_v_p[i * 15], 
                                    &pysh_v_p[i * 2 * 5 * 5], 
                                    &rot_pysh_v_p[i * 2 * 5 * 5], 
                                    &angles_v_p[i * 3], 
                                    &dj_o4[0][0][0], 
                                    num_of_dir_p,
                                    4,
                                    0)
                )
            );
        
        ceres::Solve(options2, problem2, &parameters[i * 12], &summary2);
        
        if (summary.final_cost > 0.4) {
            //std::cout << summary.FullReport() << "\n";
        }

        loss_p[i] = summary2.final_cost;
    }
}

void minimize_watson_mult_iterative2_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* dj_p, double* loss_p, int amount, int lmax_p, int num_of_dir_p) {
    // adaptive amount of directions depending on the loss
    
    // set amount of threads
    int nProcessors = omp_get_max_threads();
    //std::cout<<nProcessors<<std::endl;
    omp_set_num_threads(nProcessors);

    #pragma omp parallel for 
    for (int i=0; i<amount; i++) {
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = false;//true;
        options.logging_type = ceres::SILENT;

        ceres::GradientProblemSolver::Summary summary;

        constexpr int kNumParameters = 4;

        ceres::GradientProblem problem(
            new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters>(
                new WatsonSHApprox( &signal_p[i * 15], 
                                    &est_signal_p[i * 15], 
                                    &dipy_v_p[i * 15], 
                                    &pysh_v_p[i * 2 * 5 * 5], 
                                    &rot_pysh_v_p[i * 2 * 5 * 5], 
                                    &angles_v_p[i * 3], 
                                    &dj_o4[0][0][0], 
                                    1,
                                    4,
                                    0)
                )
            );
        
        ceres::Solve(options, problem, &parameters[i * 12], &summary);

        if (summary.final_cost < 0.01) {
            // set weight of second and third direction to zero
            parameters[i * 12 + 4] = 0;
            parameters[i * 12 + 8] = 0;

            loss_p[i] = summary.final_cost;
        } else {

            // loss after first direction too high, so continue with second direction

            ceres::GradientProblemSolver::Options options2;
            options2.minimizer_progress_to_stdout = false;//true;
            options2.logging_type = ceres::SILENT;

            ceres::GradientProblemSolver::Summary summary2;

            constexpr int kNumParameters2 = 8;

            ceres::GradientProblem problem2(
                new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters2>(
                    new WatsonSHApprox( &signal_p[i * 15], 
                                        &est_signal_p[i * 15], 
                                        &dipy_v_p[i * 15], 
                                        &pysh_v_p[i * 2 * 5 * 5], 
                                        &rot_pysh_v_p[i * 2 * 5 * 5], 
                                        &angles_v_p[i * 3], 
                                        &dj_o4[0][0][0], 
                                        2,
                                        4,
                                        0)
                    )
                );

            ceres::Solve(options2, problem2, &parameters[i * 12], &summary2);

            if (summary2.final_cost < 0.01) {
                // set weight of third direction to zero
                parameters[i * 12 + 8] = 0;

                loss_p[i] = summary2.final_cost;
            } else {

                // loss after first direction too high, so continue with second direction

                ceres::GradientProblemSolver::Options options3;
                options3.minimizer_progress_to_stdout = false;//true;
                options3.logging_type = ceres::SILENT;

                ceres::GradientProblemSolver::Summary summary3;

                constexpr int kNumParameters3 = 12;

                ceres::GradientProblem problem3(
                    new ceres::NumericDiffFirstOrderFunction<WatsonSHApprox, ceres::CENTRAL, kNumParameters3>(
                        new WatsonSHApprox( &signal_p[i * 15], 
                                            &est_signal_p[i * 15], 
                                            &dipy_v_p[i * 15], 
                                            &pysh_v_p[i * 2 * 5 * 5], 
                                            &rot_pysh_v_p[i * 2 * 5 * 5], 
                                            &angles_v_p[i * 3], 
                                            &dj_o4[0][0][0], 
                                            3,
                                            4,
                                            0)
                        )
                    );

                ceres::Solve(options3, problem3, &parameters[i * 12], &summary3);

                loss_p[i] = summary3.final_cost;
            }
        }
    }
}