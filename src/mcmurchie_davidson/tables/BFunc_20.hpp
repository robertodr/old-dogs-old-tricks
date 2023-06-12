/* This file was autogenerated on 2023-06-11T16:57 DO NOT EDIT! */

#pragma once

#include <array>

namespace mcmurchie_davidson {
/** Pretabulated coefficients for the seven-term Taylor expansion of the Boys function of order 20.
 *  See: McMurchie, L. E.; Davidson, E. R. J. Comput. Phys. 1978, 26, 218. https://doi.org/10.1016/0021-9991(78)90092-X
 *  Row i contains the coefficients for the expansion at point i. The
 *  coefficients are in natural order: from 0-th to 6-th power.
 */
template <>
inline constexpr std::array<std::array<double, 7>, 121>
pretabulated<20>()
{
    // clang-format off
  return {{{2.43902439024390243902e-2, -2.32558139534883720930e-2, 1.11111111111111111111e-2, -3.54609929078014184397e-3, 8.50340136054421768707e-4, -1.63398692810457516340e-4, 2.62054507337526205451e-5},
           {2.21723109447861978227e-2, -2.11366535013732884107e-2, 1.00967063077296737711e-2, -3.22179138309250595942e-3, 7.72448332526465362285e-4, -1.48409208965154948293e-4, 2.37982002138099524757e-5},
           {2.01564666505587783418e-2, -1.92109489873201061405e-2, 9.17506672097842188029e-3, -2.92718825419840435187e-3, 7.01701521455022386893e-4, -1.34796586523429259277e-4, 2.16123487687681746375e-5},
           {1.83242640742870867288e-2, -1.74610106067544755548e-2, 8.33769617393715495313e-3, -2.65956527017391537726e-3, 6.37443438008772496860e-4, -1.22434200230537161384e-4, 1.96275109471519726596e-5},
           {1.66589429494432735013e-2, -1.58707686144187488930e-2, 7.57687774022934319024e-3, -2.41644803854199671095e-3, 5.79078168396033243423e-4, -1.11207083313497249294e-4, 1.78251802829234330661e-5},
           {1.51452752306940052751e-2, -1.44256247458207937512e-2, 6.88560217883040989739e-3, -2.19558939701724499527e-3, 5.26064593592908682704e-4, -1.01010852938560678868e-4, 1.61885559324416100700e-5},
           {1.37694248008225442886e-2, -1.31123172830815426515e-2, 6.25750294936660515083e-3, -1.99494852069001136693e-3, 4.77911345158221908223e-4, -9.17507348058513864031e-5, 1.47023853215092773427e-5},
           {1.25188200416170933998e-2, -1.19187985106365985227e-2, 5.68679720213007087321e-3, -1.81267195241628696969e-3, 4.34172225892363926412e-4, -8.33406777262532083399e-5, 1.33528213226786778738e-5},
           {1.13820380876094896794e-2, -1.08341234217296591980e-2, 5.16823219286054267446e-3, -1.64707637919033313481e-3, 3.94442052459697558263e-4, -7.57025498717854485736e-5, 1.21272926198830796330e-5},
           {1.03486996901281935735e-2, -9.84834864147602143114e-3, 4.69703662301937385770e-3, -1.49663299362449682753e-3, 3.58352881053449205846e-4, -6.87654091585979495476e-5, 1.10143860416822685506e-5},
           {9.40937371771591833658e-3, -8.95244062745517536761e-3, 4.26887645228253304747e-3, -1.35995329449880458274e-3, 3.25570580775426572390e-4, -6.24648409185804732871e-5, 1.00037397571320369204e-5},
           {8.55549040847624307771e-3, -8.13819229520290968183e-3, 3.87981477173763482716e-3, -1.23577619381117593741e-3, 2.95791722664242738485e-4, -5.67423566479919821537e-5, 9.08594633057533425499e-6},
           {7.77926257119075474644e-3, -7.39814729442451069624e-3, 3.52627536417747435710e-3, -1.12295630998406970394e-3, 2.68740755265721659520e-4, -5.15448481954397467867e-5, 8.25246472447287955979e-6},
           {7.07361400584753056281e-3, -6.72553123297543395398e-3, 3.20500961229443562250e-3, -1.02045333797991114380e-3, 2.44167440325678908666e-4, -4.68240922723905315635e-5, 7.49554042362043580728e-6},
           {6.43211448123182094605e-3, -6.11418921032078847150e-3, 2.91306644681919076761e-3, -9.27322397150042517129e-4, 2.21844524623594524949e-4, -4.25363006420856088353e-5, 6.80813293053448107689e-6},
           {5.84892066807593882793e-3, -5.55852908089457704854e-3, 2.64776505500617176728e-3, -8.42705266784759339689e-4, 2.01565626178782476364e-4, -3.86417117717177352075e-5, 6.18384995114941536377e-6},
           {5.31872248151516249792e-3, -5.05346992108322672077e-3, 2.40667009561303405982e-3, -7.65822427631132531672e-4, 1.83143315069322985388e-4, -3.51042201220533783617e-5, 5.61688765291290421639e-6},
           {4.83669433676529435945e-3, -4.59439522195366836788e-3, 2.18756918989311110318e-3, -6.95965835178695294352e-4, 1.66407370927175403061e-4, -3.18910396020188790698e-5, 5.10197643448568137058e-6},
           {4.39845086749311199592e-3, -4.17711037378643273293e-3, 1.98845247933752012487e-3, -6.32492357351405643136e-4, 1.51203200827661286137e-4, -2.89723980364426558960e-5, 4.63433169808742336827e-6},
           {4.00000669771804345798e-3, -3.79780404836967859852e-3, 1.80749406016594924701e-3, -5.74817815451774500810e-4, 1.37390402793480844845e-4, -2.63212597861809680718e-5, 4.20960916257037874263e-6},
           {3.63773989564746035377e-3, -3.45301312123331288761e-3, 1.64303512205247765482e-3, -5.22411572837931138802e-4, 1.24841461496707920378e-4, -2.39130739239914124304e-5, 3.82386429799126181318e-6},
           {3.30835877195921872901e-3, -3.13959080889193199726e-3, 1.49356863444895628448e-3, -4.74791620929532957700e-4, 1.13440563979627564301e-4, -2.17255456092463268601e-5, 3.47351550119235491171e-6},
           {3.00887171602092981515e-3, -2.85467772602825218403e-3, 1.35772543828193413031e-3, -4.31520116781827604846e-4, 1.03082524338461700968e-4, -1.97384285221536743301e-5, 3.15531066706018035401e-6},
           {2.73655979166481091655e-3, -2.59567559466382120209e-3, 1.23426161388485284252e-3, -3.92199330682358696543e-4, 9.36718073335279701602e-5, -1.79333364156254588249e-5, 2.86629684203627309040e-6},
           {2.48895183968139525887e-3, -2.36022336198432819768e-3, 1.12204700790767444678e-3, -3.56467966051328844640e-4, 8.51216418147769166751e-5, -1.62935720221618371797e-5, 2.60379267541030166120e-6},
           {2.26380185739294068070e-3, -2.14617550584237076067e-3, 1.02005481273232138581e-3, -3.23997817400340198512e-4, 7.73532156916433954585e-5, -1.48039717157898489535e-5, 2.36536341020420977816e-6},
           {2.05906844673377827787e-3, -1.95158232725982849418e-3, 9.27352101715270315044e-4, -2.94490735257708783137e-4, 7.02949449386222781485e-5, -1.34507644767403816780e-5, 2.14879817930484454683e-6},
           {1.87289614139621437769e-3, -1.77467204768427550682e-3, 8.43091232469829216650e-4, -2.67675869831325119290e-4, 6.38818098200925020558e-5, -1.22214438405541554197e-5, 1.95208939414653561860e-6},
           {1.70359844097483299936e-3, -1.61383454549111752564e-3, 7.66502038473228252714e-4, -2.43307167778947588694e-4, 5.80547521461705586161e-5, -1.11046516349385195505e-5, 1.77341403288835408399e-6},
           {1.54964239582075849477e-3, -1.46760658142136676992e-3, 6.96884736613069414105e-4, -2.21161098815203326475e-4, 5.27601279416679578367e-5, -1.00900724180953376536e-5, 1.61111665285830988882e-6},
           {1.40963460064729741161e-3, -1.33465837644588603342e-3, 6.33603484942435327479e-4, -2.01034591026535899025e-4, 4.79492104279114148284e-5, -9.16833763244435950585e-6, 1.46369396821657657747e-6},
           {1.28230846794264103164e-3, -1.21378141807911899865e-3, 5.76080530955191353476e-4, -1.82743155709932003812e-4, 4.35777386871174959621e-5, -8.33093857861886440933e-6, 1.32978084847439450589e-6},
           {1.16651266406482545523e-3, -1.10387738254557644935e-3, 5.23790896179189774742e-4, -1.66119184316690680437e-4, 3.96055078056029984546e-5, -7.57014739716531588517e-6, 1.20813760683280741461e-6},
           {1.06120060162882791019e-3, -1.00394807053666543402e-3, 4.76257547866414132111e-4, -1.51010401685285950866e-4, 3.59959966793604321062e-5, -6.87894532031303582485e-6, 1.09763845940148064252e-6},
           {9.65420891545132461506e-4, -9.13086263680060010450e-4, 4.33047013082099357863e-4, -1.37278461202522443973e-4, 3.27160300170668450642e-5, -6.25095752419680093521e-6, 9.97261047336273996402e-7},
           {8.78308666923932916323e-4, -8.30467417366113772395e-4, 3.93765394601745778117e-4, -1.24797668853301687610e-4, 2.97354713947180183346e-5, -5.68039397365468861739e-6, 9.06076923898462385345e-7},
           {7.99077699101498838651e-4, -7.55342113315130708751e-4, 3.58054751754035335137e-4, -1.13453824318765521999e-4, 2.70269445057833804669e-5, -5.16199570776475266784e-6, 8.23242917482210402725e-7},
           {7.27013233349803442325e-4, -6.87029202297647408754e-4, 3.25589812733749429854e-4, -1.03143168371578952203e-4, 2.45655800137720740580e-5, -4.69098606515384682525e-6, 7.47993289865278231405e-7},
           {6.61465478465018009296e-4, -6.24909573802655770279e-4, 2.96074987983463606966e-4, -9.37714268058371805750e-5, 2.23287856528537825645e-5, -4.26302639428782890899e-6, 6.79632616388151331133e-7},
           {6.01843690456521076756e-4, -5.68420495245258359206e-4, 2.69241657034728144090e-4, -8.52529420367774317530e-5, 2.02960374389256882296e-5, -3.87417583586780453676e-6, 6.17529321528955740461e-7},
           {5.47610796031468196471e-4, -5.17050468569505696989e-4, 2.44845703734662331879e-4, -7.75098843205304513862e-5, 1.84486900502887435577e-5, -3.52085480250564032207e-6, 5.61109809479230722659e-7},
           {4.98278506541522487813e-4, -4.70334556883072790658e-4, 2.22665277086030698184e-4, -7.04715352841775888749e-5, 1.67698046157360735903e-5, -3.19981181539064696631e-6, 5.09853134896601331067e-7},
           {4.53402877574094170365e-4, -4.27850138102401115429e-4, 2.02498757019378483124e-4, -6.40736371283010533928e-5, 1.52439923100357425944e-5, -2.90809338901224661523e-6, 4.63286164067007210909e-7},
           {4.12580273472239764197e-4, -3.89213046530338083358e-4, 1.84162906314163694757e-4, -5.82578014742986526617e-5, 1.38572723040276096610e-5, -2.64301668346467591913e-6, 4.20979181298916916613e-7},
           {3.75443699793225906490e-4, -3.54074066869752802564e-4, 1.67491191609712064461e-4, -5.29709723826827109439e-5, 1.25969427502276840508e-5, -2.40214466969632671245e-6, 3.82541899538049664234e-7},
           {3.41659470100165517208e-4, -3.22115748429386758705e-4, 1.52332258012295669375e-4, -4.81649385715611911933e-5, 1.14514636061942101268e-5, -2.18326357652045000273e-6, 3.47619837972661885779e-7},
           {3.10924176555425952652e-4, -2.93049510232486882209e-4, 1.38548543226269037058e-4, -4.37958903212070277443e-5, 1.04103502080923876862e-5, -1.98436240949552816353e-6, 3.15891032831981373369e-7},
           {2.82961936577138953655e-4, -2.66613010422007891843e-4, 1.26015018428218903822e-4, -3.98240169653169586968e-5, 9.46407660693066791861e-6, -1.80361435111227278362e-6, 2.87063050696111982088e-7},
           {2.57521890356957842017e-4, -2.42567755793253621559e-4, 1.14618044275513993112e-4, -3.62131412461149835517e-5, 8.60398777076556745230e-6, -1.63935986927106974502e-6, 2.60870276463962430495e-7},
           {2.34375926341667155991e-4, -2.20696929498367640579e-4, 1.04254331505380555003e-4, -3.29303871523792384963e-5, 7.82221983863348536211e-6, -1.49009137696311879889e-6, 2.37071450693018147803e-7},
           {2.13316613874672302009e-4, -2.00803416977608632267e-4, 9.48299965475846699412e-5, -2.99458781699523638072e-5, 7.11162768683786008215e-6, -1.35443930052987199418e-6, 2.15447433355185841923e-7},
           {1.94155324095131889784e-4, -1.82708011998505853333e-4, 8.62597034519657878933e-5, -2.72324631562299240373e-5, 6.46571913619767435494e-6, -1.23115942700393537500e-6, 1.95799173168360011498e-7},
           {1.76720521920120432523e-4, -1.66247786342706941210e-4, 7.84658842295967980194e-5, -2.47654673061366451339e-5, 5.87859519058392720255e-6, -1.11912141295345667323e-6, 1.77945863583649696808e-7},
           {1.60856213504688937404e-4, -1.51274608186982724954e-4, 7.13780304306616458559e-5, -2.25224659095804799762e-5, 5.34489575311110489764e-6, -1.01729834807266433599e-6, 1.61723268251372635382e-7},
           {1.46420535000520716162e-4, -1.37653795593395097182e-4, 6.49320494399679287723e-5, -2.04830788114881416249e-5, 4.85975031723369694137e-6, -9.24757276584988530458e-7, 1.46982200371328229106e-7},
           {1.33284469729227689119e-4, -1.25262892766750427069e-4, 5.90696795684630864887e-5, -1.86287836772363300537e-5, 4.41873317619910843733e-6, -8.40650588443992928410e-7, 1.33587141769337843303e-7},
           {1.21330682063157112055e-4, -1.13990557866651526209e-4, 5.37379585617442863984e-5, -1.69427463403875238962e-5, 4.01782273625869422473e-6, -7.64208200414642765170e-7, 1.21414988846065068289e-7},
           {1.10452457375738295167e-4, -1.03735552187191583059e-4, 4.88887406393838581161e-5, -1.54096666677377389402e-5, 3.65336455712995825871e-6, -6.94730454468965436271e-7, 1.10353913727921862739e-7},
           {1.00552738394762443668e-4, -9.44058214490889682780e-5, 4.44782576265085222459e-5, -1.40156385202577403530e-5, 3.32203777779042726715e-6, -6.31581667604679993091e-7, 1.00302330024522715298e-7},
           {9.15432491745096804008e-5, -8.59176607954674004730e-5, 4.04667201456235692703e-5, -1.27480225188946136587e-5, 3.02082461709003937707e-6, -5.74184273255300136823e-7, 9.11679535727465540513e-8},
           {8.33436987044584718298e-5, -7.81949558513689118016e-5, 3.68179552059371713703e-5, -1.15953304426107572012e-5, 2.74698266718702321828e-6, -5.22013499962200510611e-7, 8.28669494331361445259e-8},
           {7.58810569008203734866e-5, -7.11684929055590677002e-5, 3.34990768628371253426e-5, -1.05471201935727620653e-5, 2.49801972370800871349e-6, -4.74592537974592464273e-7, 7.53231572084120390920e-8},
           {6.90888963890546172007e-5, -6.47753319077010598350e-5, 3.04801869248144869685e-5, -9.59390036206395521961e-6, 2.27167092004801635327e-6, -4.31488148978979623478e-7, 6.84673874838271794214e-8},
           {6.29067940869262301386e-5, -5.89582365504013849165e-5, 2.77341029618389710786e-5, -8.72704351238874653264e-6, 2.06587795457906376118e-6, -3.92306678277818300197e-7, 6.22367828517779927731e-8},
           {5.72797871441193740178e-5, -5.36651562292929322343e-5, 2.52361111205332565614e-5, -7.93870739158916576309e-6, 1.87877021892643114679e-6, -3.56690432476261670330e-7, 5.65742375847196419615e-8},
           {5.21578782909338730962e-5, -4.88487551500543557112e-5, 2.29637414797982525858e-5, -7.22176333595205542128e-6, 1.70864765307979770355e-6, -3.24314389130935792069e-7, 5.14278705666393467577e-8},
           {4.74955860997474348236e-5, -4.44659842834652885789e-5, 2.08965638879206349644e-5, -6.56973121672921798705e-6, 1.55396516909639153848e-6, -2.94883207897382217540e-7, 4.67505465892335732576e-8},
           {4.32515360728695220883e-5, -4.04776922622507705370e-5, 1.90160024105706331582e-5, -5.97672032683913958336e-6, 1.41331849967396417145e-6, -2.68128515511873146450e-7, 4.24994415691514545301e-8},
           {3.93880888429156593910e-5, -3.68482716702007908477e-5, 1.73051666902140587859e-5, -5.43737566512810775065e-6, 1.28543134105794984903e-6, -2.43806439484888097432e-7, 3.86356476514199055002e-8},
           {3.58710021103181620253e-5, -3.35453374981540286570e-5, 1.57486986729023348123e-5, -4.94682912454941181942e-6, 1.16914367172226846281e-6, -2.21695367691346023870e-7, 3.51238145352776644269e-8},
           {3.26691232501720804247e-5, -3.05394348358951016852e-5, 1.43326332996059152639e-5, -4.50065513583343589440e-6, 1.06340113913840820987e-6, -2.01593913138240595010e-7, 3.19318236955861801766e-8},
           {2.97541098001199680767e-5, -2.78037731365498044735e-5, 1.30442718875257118147e-5, -4.09483035928876270919e-6, 9.67245416823708230951e-7, -1.83319065093174550120e-7, 2.90304924789042299099e-8},
           {2.71001752949721427962e-5, -2.53139846331369295078e-5, 1.18720670433403490600e-5, -3.72569705467529222327e-6, 8.79805442829127594943e-7, -1.66704509485166656793e-7, 2.63933053310699955583e-8},
           {2.46838581445748709913e-5, -2.30479047077889851890e-5, 1.08055180562012266419e-5, -3.38992979297100215628e-6, 8.00289458972523387149e-7, -1.51599103058126030809e-7, 2.39961696653235148107e-8},
           {2.24838114611935391458e-5, -2.09853722148214001100e-5, 9.83507581445073398088e-6, -3.08450520462823893391e-6, 7.27977777521353881679e-7, -1.37865487182169910619e-7, 2.18171941089809779225e-8},
           {2.04806119332981895111e-5, -1.91080479411591202220e-5, 8.95205637740010348876e-6, -2.80667448686855578230e-6, 6.62216208747482136570e-7, -1.25378828521777707272e-7, 1.98364870745778259537e-8},
           {1.86565860158720000985e-5, -1.73992495532984299002e-5, 8.14856241286896921931e-6, -2.55393841795604538571e-6, 6.02410088878807807345e-7, -1.14025674934683215265e-7, 1.80359737901722755036e-8},
           {1.69956518647987550878e-5, -1.58438015305160455847e-5, 7.41741178329181511937e-6, -2.32402464945304748737e-6, 5.48018853514495198602e-7, -1.03702916042340131622e-7, 1.63992300949061711957e-8},
           {1.54831755859658184104e-5, -1.44278987208209849677e-5, 6.75207262870228214446e-6, -2.11486706841254468661e-6, 4.98551106604023286890e-7, -9.43168388816988473610e-8, 1.49113314615567860992e-8},
           {1.41058404997557416011e-5, -1.31389822804362524583e-5, 6.14660435441982093560e-6, -1.92458704049252441986e-6, 4.53560139661918258427e-7, -8.57822699279130957412e-8, 1.35587158491328719546e-8},
           {1.28515282397838080322e-5, -1.19656268705387795574e-5, 5.59560398533287018541e-6, -1.75147636226502654663e-6, 4.12639860042244169633e-7, -7.80217955766475959230e-8, 1.23290591168895356102e-8},
           {1.17092106121526510180e-5, -1.08974380876165720519e-5, 5.09415739779760074987e-6, -1.59398176669664461376e-6, 3.75421090868270120677e-7, -7.09650539003165416622e-8, 1.12111618476566251962e-8},
           {1.06688512391215681562e-5, -9.92495919706332837260e-6, 4.63779498480419118589e-6, -1.45069084004300275618e-6, 3.41568208638163481740e-7, -6.45480911515927347455e-8, 1.01948465341803671814e-8},
           {9.72131609982128277840e-6, -9.03958632438057166001e-6, 4.22245135061474066105e-6, -1.32031922135915175009e-6, 3.10776087638869598612e-7, -5.87127770860218432269e-8, 9.27086417825455872848e-9},
           {8.85829216129554525760e-6, -8.23349133537298549465e-6, 3.84442866792223835912e-6, -1.20169896760069673994e-6, 2.82767323126352342015e-7, -5.34062737191166715084e-8, 8.43080943966128210551e-9},
           {8.07221336645721649901e-6, -7.49955170671170682627e-6, 3.50036336405754536572e-6, -1.09376797798560986675e-6, 2.57289707797292094426e-7, -4.85805526269454114720e-8, 7.66704355116495146152e-9},
           {7.35619331217816095262e-6, -6.83128675184435975185e-6, 3.18719583318970507604e-6, -9.95560381002827615723e-7, 2.34113938408952921245e-7, -4.41919563475518544959e-8, 6.97262428774512647619e-9},
           {6.70396401129807819621e-6, -6.22279962503365067936e-6, 2.90214289910377254756e-6, -9.06197796280687607973e-7, 2.13031531521834622670e-7, -4.02007998477211640784e-8, 6.34124234358379143175e-9},
           {6.10982018739090654877e-6, -5.66872457883797947943e-6, 2.64267277825438586260e-6, -8.24881391547499626263e-7, 1.93852929263532506350e-7, -3.65710083894107313258e-8, 5.76716352964921831931e-9},
           {5.56856860117086037052e-6, -5.16417899808942961484e-6, 2.40648231561325997324e-6, -7.50884662202059744223e-7, 1.76405777759755132173e-7, -3.32697884660386638433e-8, 5.24517625859419141498e-9},
           {5.07548195291067361350e-6, -4.70471977681423560559e-6, 2.19147628656474297254e-6, -6.83546867630976726517e-7, 1.60533362465865886114e-7, -3.02673287838781782929e-8, 4.77054383261211710564e-9},
           {4.62625694659689857359e-6, -4.28630364397562750437e-6, 1.99574857694603186503e-6, -6.22267064423318919741e-7, 1.46093186074312023286e-7, -2.75365285408766062499e-8, 4.33896109432603401410e-9},
           {4.21697613913857753074e-6, -3.90525107976015172179e-6, 1.81756507045085859050e-6, -5.66498682096760471745e-7, 1.32955675983223565017e-7, -2.50527505068731803071e-8, 3.94651504113593980352e-9},
           {3.84407323211877059748e-6, -3.55821349670402320615e-6, 1.65534808817468276663e-6, -5.15744591913500538045e-7, 1.21003009501416606053e-7, -2.27935966377648408136e-8, 3.58964904009453533973e-9},
           {3.50430149464480994464e-6, -3.24214338956402230350e-6, 1.50766223921859983882e-6, -4.69552623874471472684e-7, 1.10128046046026923300e-7, -2.07387041637828262607e-8, 3.26513031366114503508e-9},
           {3.19470503409574006415e-6, -2.95426718474860708255e-6, 1.37320155411811277569e-6, -4.27511491078318575526e-7, 1.00233356571017523096e-7, -1.88695602806205181541e-8, 2.97002039690758173359e-9},
           {2.91259265724134235802e-6, -2.69206054458541755176e-6, 1.25077778453952995982e-6, -3.89247084355057978777e-7, 9.12303413569256515181e-8, -1.71693337434416383914e-8, 2.70164829419905882463e-9},
           {2.65551408755016969982e-6, -2.45322590393473316934e-6, 1.13930976329820577391e-6, -3.54419103467485949141e-7, 8.30384281026801624346e-8, -1.56227218193851904006e-8, 2.45758608830284969766e-9},
           {2.42123832572654315202e-6, -2.23567203686787401557e-6, 1.03781372839644453041e-6, -3.22717994247466336504e-7, 7.55843429966146712293e-8, -1.42158111955105871530e-8, 2.23562677751890834485e-9},
           {2.20773395981141531589e-6, -2.03749546949958760465e-6, 9.45394523543082922793e-7, -2.93862163827451686487e-7, 6.88014481127213370632e-8, -1.29359515675017184524e-8, 2.03376413699034185896e-9},
           {2.01315124872495201911e-6, -1.85696357176186829411e-6, 8.61237595581866436589e-7, -2.67595448665685832483e-7, 6.26291390859119569611e-8, -1.17716407510611601556e-8, 1.85017441902837635877e-9},
           {1.83580581907888491916e-6, -1.69249917608578333805e-6, 7.84601716494733153282e-7, -2.43684812369839156264e-7, 5.70122975721806290541e-8, -1.07124202638543531463e-8, 1.68319972424966741704e-9},
           {1.67416382958916538377e-6, -1.54266658475629367476e-6, 7.14812364225324083765e-7, -2.21918252419540583625e-7, 5.19007935011908237885e-8, -9.74878042208864689222e-9, 1.53133289073087567102e-9},
           {1.52682847060580863415e-6, -1.40615884024847241724e-6, 6.51255702547686258838e-7, -2.02102896792654715681e-7, 4.72490325845651362475e-8, -8.87207408322368903621e-9, 1.39320376237926933657e-9},
           {1.39252767826677581642e-6, -1.28178614425642279380e-6, 5.93373105639950715444e-7, -1.84063273230699861726e-7, 4.30155449572378428154e-8, -8.07443824571613246753e-9, 1.26756671042858546053e-9},
           {1.27010295368507234916e-6, -1.16846532149239401742e-6, 5.40656177962505694106e-7, -1.67639735451411203380e-7, 3.91626112054568573950e-8, -7.34872278883713011108e-9, 1.15328929351425428803e-9},
           {1.15849918749180736001e-6, -1.06521023375756630204e-6, 4.92642224529992026013e-7, -1.52687032045562755497e-7, 3.56559223768572998408e-8, -6.68842570113226726547e-9, 1.04934195226846013113e-9},
           {1.05675539907263633783e-6, -9.71123058353525617030e-7, 4.48910130747405258967e-7, -1.39073005093877067452e-7, 3.24642708786228148068e-8, -6.08763420562608019896e-9, 9.54788643900503289508e-10},
           {9.63996308032471594334e-7, -8.85386352692545325437e-7, 4.09076614689979122633e-7, -1.26677406720106152397e-7, 2.95592694519629137078e-8, -5.54097124395680014611e-9, 8.68778330879838907211e-10},
           {8.79424662877830529020e-7, -8.07255834046335103497e-7, 3.72792818078244453383e-7, -1.15390822868968288427e-7, 2.69150956675613567673e-8, -5.04354683076116432403e-9, 7.90537245698099049452e-10},
           {8.02314258685214690852e-7, -7.36053809811263040232e-7, 3.39741205264506817018e-7, -1.05113694572421198868e-7, 2.45082596196529479668e-8, -4.59091383426736824362e-9, 7.19361860825143575799e-10},
           {7.32003581688765698125e-7, -6.71163199521803893730e-7, 3.09632742332932700081e-7, -9.57554278536680606742e-8, 2.23173927081062922838e-8, -4.17902777960872602122e-9, 6.54612499458836572093e-10},
           {6.67890024326950708920e-7, -6.12022095166237865539e-7, 2.82204330947923580554e-7, -8.72335842234513928617e-8, 2.03230555902753774465e-8, -3.80421030821524819949e-9, 5.95707528558783637291e-10},
           {6.09424619386816852441e-7, -5.58118811197831373048e-7, 2.57216473887571878332e-7, -7.94731444549628291232e-8, 1.85075635592017143310e-8, -3.46311596011530538378e-9, 5.42118081005233424730e-10},
           {5.56107246521555015298e-7, -5.08987380034761575949e-7, 2.34451151291744621406e-7, -7.24058389887852225103e-8, 1.68548277636078836379e-8, -3.15270197639315443072e-9, 4.93363258585216011128e-10},
           {5.07482268634564324787e-7, -4.64203452842878054938e-7, 2.13709888556800353662e-7, -6.59695389237760044362e-8, 1.53502108294816826857e-8, -2.87020084668098301436e-9, 4.49005771923711973401e-10},
           {4.63134559459029666023e-7, -4.23380569033292758444e-7, 1.94811998538443151668e-7, -6.01077020992245386607e-8, 1.39803955742339977323e-8, -2.61309535167074495294e-9, 4.08647977489052230136e-10},
           {4.22685887150804519314e-7, -3.86166761214761841608e-7, 1.77592982296521414485e-7, -5.47688692730011741764e-8, 1.27332656236267309454e-8, -2.37909587344283178616e-9, 3.71928275445902457405e-10},
           {3.85791621885609176349e-7, -3.52241465348853222611e-7, 1.61903074044992558156e-7, -4.99062058543240541735e-8, 1.15977968500041322169e-8, -2.16611976713552810242e-9, 3.38517835439759052065e-10},
           {3.52137738337826731867e-7, -3.21312708591229512897e-7, 1.47605917269815828862e-7, -4.54770850623514532947e-8, 1.05639586488156018232e-8, -1.97227260631215348019e-9, 3.08117620404454885547e-10},
           {3.21438086543503392700e-7, -2.93114549789803423967e-7, 1.34577360159024702070e-7, -4.14427087568323909866e-8, 9.62262415989300701179e-9, -1.79583113149507442502e-9, 2.80455681216521570636e-10}}};
    // clang-format on
}
}  // namespace mcmurchie_davidson