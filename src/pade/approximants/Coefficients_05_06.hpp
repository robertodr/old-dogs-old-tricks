/* This file was autogenerated on 2023-06-11T16:37 DO NOT EDIT! */

#pragma once

#include <array>

#include "HornerScheme.hpp"

namespace pade {
/**@{ [5/6] Padé approximant coefficients for Boys function of orders 1 <= O <= 20.*/

template <>
inline constexpr auto
approximant<5, 6, 1ul>(double* ys, double x, size_t offset) -> void
{
    ys[1 + offset] = horner(x,
                            3.33333333333333333333e-1,
                            -3.31141637140582770591e-2,
                            9.29010265076453380716e-3,
                            -4.76167949070718654664e-4,
                            3.59642466908522207376e-5,
                            -6.94478009668775785250e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.00657508857825168823e-1,
                            1.13979098981274417001e-1,
                            1.52307591990026564729e-2,
                            1.27292530465022280432e-3,
                            6.38968500080321981476e-5,
                            1.52978069273440939007e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 2ul>(double* ys, double x, size_t offset) -> void
{
    ys[2 + offset] = horner(x,
                            2.00000000000000000000e-1,
                            -4.23307804242538800471e-2,
                            6.74270376013519377503e-3,
                            -5.60745293819705978998e-4,
                            2.93220398733470454631e-5,
                            -7.10544496138912273952e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.02631812164444885479e-1,
                            1.14958463997501680725e-1,
                            1.54470900346643727086e-2,
                            1.29986583501452262928e-3,
                            6.58198642241591593674e-5,
                            1.59410853251796836551e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 3ul>(double* ys, double x, size_t offset) -> void
{
    ys[3 + offset] = horner(x,
                            1.42857142857142857143e-1,
                            -3.90547927961932974230e-2,
                            5.96000701528491395920e-3,
                            -5.30092263884054060883e-4,
                            2.77459073745888883145e-5,
                            -6.82289068988269953344e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.04394228204424695817e-1,
                            1.15844852861950979309e-1,
                            1.56454235118644711088e-2,
                            1.32486241301952852717e-3,
                            6.76231704967965844175e-5,
                            1.65493333625906394027e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 4ul>(double* ys, double x, size_t offset) -> void
{
    ys[4 + offset] = horner(x,
                            1.11111111111111111111e-1,
                            -3.46681553747325667873e-2,
                            5.41692381766337547569e-3,
                            -4.94967099387826083705e-4,
                            2.63132904103163729135e-5,
                            -6.54603363678303372621e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.06168419809225080732e-1,
                            1.16736266230853836943e-1,
                            1.58446413394484283664e-2,
                            1.34992798330289974574e-3,
                            6.94267563469838336348e-5,
                            1.71552481439689168453e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 5ul>(double* ys, double x, size_t offset) -> void
{
    ys[5 + offset] = horner(x,
                            9.09090909090909090909e-2,
                            -3.07477839418057234341e-2,
                            4.95464988022358383964e-3,
                            -4.62310314656899196775e-4,
                            2.49377192376885270558e-5,
                            -6.27134786221473493723e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.07928222793983196071e-1,
                            1.17619901303009306091e-1,
                            1.60419073076593362145e-2,
                            1.37470536406335943452e-3,
                            7.12050969915495576226e-5,
                            1.77506078051224037777e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 6ul>(double* ys, double x, size_t offset) -> void
{
    ys[6 + offset] = horner(x,
                            7.69230769230769230769e-2,
                            -2.74623386628047602154e-2,
                            4.54901983427717449404e-3,
                            -4.32286835420208831879e-4,
                            2.36265747882117415666e-5,
                            -6.00112237224316572738e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.09656264050204783867e-1,
                            1.18486412179310159538e-1,
                            1.62350111989632909211e-2,
                            1.39890630914558357557e-3,
                            7.29373611268347664629e-5,
                            1.83286924390785698212e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 7ul>(double* ys, double x, size_t offset) -> void
{
    ys[7 + offset] = horner(x,
                            6.66666666666666666667e-2,
                            -2.47348815051372757032e-2,
                            4.19252233768612790612e-3,
                            -4.04777637697133512807e-4,
                            2.23841494112361691236e-5,
                            -5.73741287552041177292e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.11329718599411452688e-1,
                            1.19324274077156513069e-1,
                            1.64214002733838591896e-2,
                            1.42221843897593426266e-3,
                            7.46022301548919499906e-5,
                            1.88829276115867451784e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 8ul>(double* ys, double x, size_t offset) -> void
{
    ys[8 + offset] = horner(x,
                            5.88235294117647058824e-2,
                            -2.24593020454301751391e-2,
                            3.87938304449271951013e-3,
                            -3.79682292941380102318e-4,
                            2.12152873358129305917e-5,
                            -5.48255310142315552522e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.12928707332950180529e-1,
                            1.20123818818690052346e-1,
                            1.65990125414110064135e-2,
                            1.44439864633045579144e-3,
                            7.61836875390852769535e-5,
                            1.94085259707658595885e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 9ul>(double* ys, double x, size_t offset) -> void
{
    ys[9 + offset] = horner(x,
                            5.26315789473684210526e-2,
                            -2.05432246253575127712e-2,
                            3.60401505932845038287e-3,
                            -3.56868445193377384172e-4,
                            2.01230928457646420789e-5,
                            -5.23861413025183722487e-7) /
                     horner(x,
                            1.00000000000000000000e+0,
                            5.14440636880112019252e-1,
                            1.20879098376948533285e-1,
                            1.67666194857236789859e-2,
                            1.46530712089816510904e-3,
                            7.76728770376331829008e-5,
                            1.99029545304537390476e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 10ul>(double* ys, double x, size_t offset) -> void
{
    ys[10 + offset] = horner(x,
                             4.76190476190476190476e-2,
                             -1.89135054798883799773e-2,
                             3.36119530434125205887e-3,
                             -3.36166936295403608731e-4,
                             1.91078799890297427619e-5,
                             -5.00701683527616069227e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.15859863183213585694e-1,
                             1.21587585167143914957e-1,
                             1.69237331048252397721e-2,
                             1.48489306088538517887e-3,
                             7.90669576583848630597e-5,
                             2.03655406057537194304e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 11ul>(double* ys, double x, size_t offset) -> void
{
    ys[11 + offset] = horner(x,
                             4.34782608695652173913e-2,
                             -1.75136588242920159562e-2,
                             3.14626821725554626556e-3,
                             -3.17388698014813701508e-4,
                             1.81675151081551456807e-5,
                             -4.78848620554848534151e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.17185847041283633008e-1,
                             1.22249222348932580549e-1,
                             1.70703917758112415910e-2,
                             1.50316801190251575173e-3,
                             8.03672502331872986459e-5,
                             2.07968875907117326339e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 12ul>(double* ys, double x, size_t offset) -> void
{
    ys[12 + offset] = horner(x,
                             4.00000000000000000000e-2,
                             -1.63001839958202320862e-2,
                             2.95521026181555901811e-3,
                             -3.00341629529212326083e-4,
                             1.72982003658422777198e-5,
                             -4.58317542850259093358e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.18421326030420123772e-1,
                             1.22865520111231363364e-1,
                             1.72069637691093015644e-2,
                             1.52018215053926128879e-3,
                             8.15776279203353923018e-5,
                             2.11983751253752425258e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 13ul>(double* ys, double x, size_t offset) -> void
{
    ys[13 + offset] = horner(x,
                             3.70370370370370370370e-2,
                             -1.52393910347430976430e-2,
                             2.78460381547287382625e-3,
                             -2.84841839164980150499e-4,
                             1.64952086467562600442e-5,
                             -4.39082885089273601993e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.19570924820557053295e-1,
                             1.23438879296751190203e-1,
                             1.73340022613906424370e-2,
                             1.53600699345549667949e-3,
                             8.27033531627902974819e-5,
                             2.15717994729218766970e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 14ul>(double* ys, double x, size_t offset) -> void
{
    ys[14 + offset] = horner(x,
                             3.44827586206896551724e-2,
                             -1.43049538414761149822e-2,
                             2.63157089471844722936e-3,
                             -2.70719655244041824763e-4,
                             1.57534292533632634200e-5,
                             -4.21092668373262579113e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.20640209564934601001e-1,
                             1.23972135178156976649e-1,
                             1.74521483141365651183e-2,
                             1.55072390311743593038e-3,
                             8.37503080729690621846e-5,
                             2.19191359064377143730e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 15ul>(double* ys, double x, size_t offset) -> void
{
    ys[15 + offset] = horner(x,
                             3.22580645161290322581e-2,
                             -1.34760921003434492302e-2,
                             2.49369594279145867559e-3,
                             -2.57822051943443747127e-4,
                             1.50677275301704468409e-5,
                             -4.04279577924900989807e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.21635084283292467804e-1,
                             1.24468268120364073980e-1,
                             1.75620697087789639867e-2,
                             1.56441687796878765842e-3,
                             8.47245133920406815607e-5,
                             2.22423903667838723377e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 16ul>(double* ys, double x, size_t offset) -> void
{
    ys[16 + offset] = horner(x,
                             3.03030303030303030303e-2,
                             -1.27362338894791196328e-2,
                             2.36895161689976106056e-3,
                             -2.46012934216976083442e-4,
                             1.44331615400854580845e-5,
                             -3.88568791839449288826e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.22561424504331909261e-1,
                             1.24930229087259112070e-1,
                             1.76644242847369525240e-2,
                             1.57716825472660370181e-3,
                             8.56318430068341634212e-5,
                             2.25435116415052728012e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 17ul>(double* ys, double x, size_t offset) -> void
{
    ys[17 + offset] = horner(x,
                             2.85714285714285714286e-2,
                             -1.20720308251502867497e-2,
                             2.25563308744657053320e-3,
                             -2.35172273235615438865e-4,
                             1.38451000868886780520e-5,
                             -3.73883219854222369763e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.23424867065685909705e-1,
                             1.25360840350762516651e-1,
                             1.77598392534822860424e-2,
                             1.58905630354857560722e-3,
                             8.64778654151461422100e-5,
                             2.28243428831137683349e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 18ul>(double* ys, double x, size_t offset) -> void
{
    ys[18 + offset] = horner(x,
                             2.70270270270270270270e-2,
                             -1.14726283602552783717e-2,
                             2.15230222526165402042e-3,
                             -2.25194704754913225026e-4,
                             1.32992768174735266494e-5,
                             -3.60146845136051900554e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.24230699388503418197e-1,
                             1.25762743918395823774e-1,
                             1.78489006168595436694e-2,
                             1.60015401763728396696e-3,
                             8.72677652329620459029e-5,
                             2.30865979647021952405e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 19ul>(double* ys, double x, size_t offset) -> void
{
    ys[19 + offset] = horner(x,
                             2.56410256410256410256e-2,
                             -1.09291205565296053050e-2,
                             2.05774121632051017141e-3,
                             -2.15987944188501900627e-4,
                             1.27918048788866156316e-5,
                             -3.47286742258152227143e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.24983810490467344324e-1,
                             1.26138379468555334253e-1,
                             1.79321488137410006404e-2,
                             1.61052863736311880386e-3,
                             8.80063138697896990287e-5,
                             2.33318531055588402304e-6);
}

template <>
inline constexpr auto
approximant<5, 6, 20ul>(double* ys, double x, size_t offset) -> void
{
    ys[20 + offset] = horner(x,
                             2.43902439024390243902e-2,
                             -1.04341388491887654961e-2,
                             1.97091445246844639071e-3,
                             -2.07471210486771660410e-4,
                             1.23191684335274851197e-5,
                             -3.35234201059804127445e-7) /
                      horner(x,
                             1.00000000000000000000e+0,
                             5.25688679276283870474e-1,
                             1.26489980026526064823e-1,
                             1.80100780966257784433e-2,
                             1.62024161283202233174e-3,
                             8.86978694911352423539e-5,
                             2.35615475902650653338e-6);
}

/**@}*/
}  // namespace pade
