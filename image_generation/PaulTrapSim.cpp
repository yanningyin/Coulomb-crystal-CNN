//Run make PaulTrapSim to compile this code

#include <gtk/gtk.h>
#include <cairo.h>
#include "OpenMM.h"
#include <cstdio>
#include <math.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <deque>
#include <random>
#include <stdlib.h>
#include <iomanip>
using namespace std;


//define the temperature control parameters
bool isControlTemp = true;
int targetT = 40; //[mK], can be overwriten by input arguments
double currentT = 0; //[mK]
double tolReachT = 1; //[mK] // torlerance to regard temperature as stabilized
// PID parameters (play with these to get the most stable temperature)
double kp_begin = 0.03;//0.01;
double kp_stable = 0.646875;//0.45/5;
double ki_stable = 0.0;
double kd_stable = 0;//0.15/5;
double kp = 0;
double ki = 0;
double kd = 0;
// Control variables
double integral = 0.0;
double lastError = 0.0;
double correction = 0.0;

//definition of the rf frequency
double rffreq = 10e6;

//these parameters are used if the isLinear flag below is set to 1
double kappa = 0.165;
double z0 = 220e-6;
double r0 = 282e-6;

double VRF = 300; // RF amplitude
double VDC = 2.0;

//otherwise the trap is taken to be a generic Paul trap with qx,qy,qz etc set below

double ax=-4.62e-4;
double ay=-4.10e-4;
double az= 8.83e-4;

double qx= 8.73e-2;
double qy=-9.15e-2;
double qz= 0;


int isLinear = 2; //a flag to determine whether to enable the special linear functions.
int coolingAxis = 2; //determines which axis the laser cooling works on. 0 is x, 1 is y, 2 is z.

int maxDC = 500.0; // maximum RF amplitude and DC values
int maxRF = 3000.0;


int stepNum = 0;
int stepsDone = 0;

double camass = 40*1.6e-27;
double charge = 1.6e-19;
double timeStep = 1000.0; //in ps for openmm
double timeStepSI = timeStep*1e-12;


//define various constants needed to calculate the secular temperature
double timestepsPerRFPeriod =  (1.0/rffreq * 1e12) / (timeStep);
int secularStepFrequency = round(timestepsPerRFPeriod);
int isSecularStep = 1;
double secularTempWorking = 0;
double secularTemp;
//averaging for secular temperature
int secularAverageRange = 100;
double averageSecularTemp = 0;
std::deque<double> secularTempAverage;
int secularAverageBufferSize=0;


double localPi = 3.141592653589793238;
double kboltz = 1.38e-23;
double hbar = 1.054e-34;
double speedOfLight = 299792458;


//define parameters for laser cooling - these are fixed
double isat = 430.524; //in W/m^2, NOT MW/M^2
double spotRadius = 200e-6;
double wavelength = 397e-9;
double myLinewidth = 2*localPi*20.7e6;
double wavevector = 2*localPi/(397*1e-9);
double lwSq = pow(myLinewidth,2);

//define laser cooling variables - these can be varied
double laserPower = 0.5; //initial laser power in mw
double detuningFM = 10; //initial laser detuning in FM
double detuning = 2*localPi * speedOfLight*( 1 / (wavelength - detuningFM*1e-15) - 1/wavelength);
double laserIntensity = laserPower*1e-3 / (localPi * spotRadius * spotRadius);

int numSteps = 0.5e-3 / (timeStep*1e-12);
int isSimRunning = 0; //if it takes a step when it shouldn't, end the callback.


//sets some default values
int numIons = 600;// can be overwriten by input arguments
double frictionCoeffSI = 5.81662578e-22;
double frictionCoeffOMM = frictionCoeffSI * 6e14;
double velocityKickSize =  0.000; //set the initial value for the velocity kicks
double bgPressure = 0.5e-9; //in mbar

double rateScalingFactor = 1e5; //large, but not too large - bgKickProb should be less than one.
double bgKickProb = 1.27936e6 * bgPressure * 100 * rateScalingFactor * timeStepSI;
double velScalingFactor = 1/sqrt(rateScalingFactor);
double velScaleFactor = velScalingFactor*1775.0; // mean speed of a H2 molecule
double m1m2Factor = 4.0/(42.0); //momentum transfer for head on pure elastic
double bgKickSize = velScaleFactor*m1m2Factor; // size of the velocity kick


//added additional heating term
double heatingKickProb = 1.0;
double heating = 0; // currently set to 0: no heating
double heatingKickSize = heating*1e-7;




//make a random number generator
//these are very slow, use drand48() where possible for numbers in the range [0,1.0)
std::default_random_engine generator;
std::uniform_real_distribution<double> realDistribution(0.0,1.0);

std::vector<int> ionLevels;
int kickTermIndex;

//openMM stuff we need to initialise at the start
//define a structure containing the system, context, and integator - means that a) they can be defined explicitly later and b) helps keep openmm stuff to openmm
struct MyOpenMMData {
    MyOpenMMData() : system(0), context(0), integrator(0) {}
    ~MyOpenMMData() {delete context; delete integrator; delete system;}
    OpenMM::System*         system;
    OpenMM::CustomIntegrator*     integrator;
    OpenMM::Context*  context;
};

MyOpenMMData* omm;

OpenMM::NonbondedForce* nonbond;
OpenMM::CustomExternalForce* paulTrapForce;
OpenMM::CustomExternalForce* frictionForce;

int systemInitialized = 0;
int freezeWaiting = 0;

OpenMM::State state;
std::vector<OpenMM::Vec3> positionList;
std::vector<OpenMM::Vec3> velocityList;
std::vector<OpenMM::Vec3> secularVelocityList;
double axialTemp = 0;

//define the histogram structure - the x,y,z sizes can be modified as needed.
int histXSize = 252;
int histYSize = 600;
int histZSize = 404;
int xbin,ybin,zbin;
unsigned short *hist;
double histRes = 1.52 * 1e3; //resolution in nm per pixel
double histTimeLength = 1000e-6; //how long to sample for a histogram in microseconds.
int histStepsLength = (int)round(histTimeLength / timeStepSI); //casts the histogram length in time to a number of steps
int histStepsDone = 0;
int doingHist = 0;

int countReachTargetT = 0;
int numLastFewTemps = 50;
std::vector<double> lastFewTemps(numLastFewTemps);
double avgT = 0;
double varianceT = 0.0;

bool isTempStabilized(){

     if(abs(currentT - targetT) > tolReachT){
         countReachTargetT = 0;
     }
     else{
         // cout << "countReachTargetT: " << countReachTargetT << endl;
         if(++countReachTargetT > 100){
             return true;
         }
     }
//    if(currentT > 1000){
//        freezeWaiting = 2;
//    }
    return false;
}


double update_pid(double T, double dt)
{
//    cout << T << endl;
    double error = targetT - T;
    kp = abs(error) < 1? kp_stable:kp_begin;
    kd = abs(error) < 1? kd_stable:0;
    ki = abs(error) < 1? ki_stable:0;
    integral += error * dt;
    double derivative = (error - lastError) / dt;
    double output = kp * error + ki * integral + kd * derivative;
    lastError = error;
    return output;
}

double avgTemp = 0;
int counterTemp = 0;

void startHist(){
    if(doingHist==0){
    	// printf("starting histogram\n");
        //outfile << "-----------Start histogram------------" << endl;
        //cout << "-----------Start histogram------------" << endl;
    	hist = new unsigned short[histXSize * histYSize * histZSize];
    	histStepsDone = 0;
    	doingHist = 1;
    }
}


//--------------------------------------------Code for a step of the simulation.
int doStep(ofstream& ofInfo){
    if(stepNum % secularStepFrequency == 0){
        isSecularStep = 1;
        secularTempWorking = 0;
    }
    else{
        isSecularStep = 0;
    }

    //openMM update - do this on every step. this is just whatever MD code is needed.
    state  = omm->context->getState(OpenMM::State::Positions|OpenMM::State::Velocities);
    positionList = state.getPositions();
    velocityList = state.getVelocities();

    if(stepNum==0){
        secularVelocityList = state.getVelocities();
    }

    double currentTime = state.getTime()* 1e-12;
    double atomSecularVelMag;
    omm->context->setParameter("rffactor", cos(rffreq*currentTime*2*localPi));
    int doFreeze = 0;

    if(freezeWaiting==2 && cos(rffreq*currentTime*2*localPi)>0.999){
        doFreeze = 1;
        freezeWaiting = 1;
    }
    if(freezeWaiting==1 && cos(rffreq*currentTime*2*localPi)<-0.999){
        doFreeze = 1;
        freezeWaiting = 0;
    }

    axialTemp=0;
    double upperPop=0;
    for (int a = 0; a < numIons; ++a)
    {

        if(doingHist==1){

            xbin = (int)floor(positionList[a][0]/histRes+histXSize/2);
            ybin = (int)floor(positionList[a][1]/histRes+histYSize/2);
            zbin = (int)floor(positionList[a][2]/histRes+histZSize/2);
            if (xbin >= 0 && xbin < histXSize && ybin >= 0 && ybin < histYSize && zbin >= 0 && zbin < histZSize) {
                hist[xbin  +  ybin * histXSize  +  zbin * histXSize * histYSize]++;
            }

        }


        if(isSecularStep==1){
            atomSecularVelMag = pow(secularVelocityList[a][0],2)+pow(secularVelocityList[a][1],2)+pow(secularVelocityList[a][2],2);
            secularTempWorking += (atomSecularVelMag * 1e6 * camass)/(kboltz * numIons*3);
            secularVelocityList[a][0]=velocityList[a][0]/secularStepFrequency;
            secularVelocityList[a][1]=velocityList[a][1]/secularStepFrequency;
            secularVelocityList[a][2]=velocityList[a][2]/secularStepFrequency;

        }
        else{
            secularVelocityList[a][0]+=velocityList[a][0]/secularStepFrequency;
            secularVelocityList[a][1]+=velocityList[a][1]/secularStepFrequency;
            secularVelocityList[a][2]+=velocityList[a][2]/secularStepFrequency;
        }

        double atomVel = velocityList[a][coolingAxis];
        //calculate an instantaneous temperature on the cooled axis.
        axialTemp += (atomVel*atomVel * 1e6 * camass)/(kboltz * numIons);
        //update the friction force based on one velocity component.

        //double newFriction = -frictionCoeffOMM*atomVel;
        //frictionForce->setParticleParameters(frictionTermIndex[a],a,{newFriction});

        //single-axis laser cooling code:
        int isUnCooled[3] = {1,1,1};
        isUnCooled[coolingAxis] = 0;
        double beamWaistSq = 2*pow(spotRadius,2);
        double radialDistSq = (pow(positionList[a][0]*isUnCooled[0],2) + pow(positionList[a][1]*isUnCooled[1],2) + pow(positionList[a][2]*isUnCooled[2],2)) * 1e-18;
        double rabiSqGaussian = exp(-2 * radialDistSq/ beamWaistSq) *laserIntensity/isat * lwSq/2; //this version turns on gaussian beam dependency
        //double rabiSqGaussian = laserIntensity/isat * lwSq / 2;
        double fliprate = myLinewidth*(rabiSqGaussian )/(lwSq + 4*pow(-detuning - wavevector*velocityList[a][coolingAxis]*1000,2));
        double xkick = 0;
        double ykick = 0;
        double zkick = 0;


        double kickDir[3] = {0,0,0};

        //using drand48() instead of the real Distribution(gener ator)

        if(ionLevels[a]==0 ){

            if(drand48()<fliprate*timeStepSI){
                ionLevels[a]=1;
                kickDir[coolingAxis] = 1;
            }
        }
        else{

            double decayProb = (fliprate+myLinewidth)*timeStepSI;

            if(decayProb > 1){
                cout << "Warning: probability of absorption cycle is greater than one, reduce timestep. Or fix this part of the force. \n";
            }


            if(drand48()<decayProb){//1+ added here to make spontaneous decay instant.


                if(drand48()  < (myLinewidth)/(fliprate+myLinewidth)    ){
                    ionLevels[a]=0;
                    double phi= drand48()*2*localPi; //map a random number from [0,1] -> [0,2Pi] for phi, [0,Pi] for theta
                    double costheta = 2*drand48() - 1; //turns out it's actually quicker + more random to generate a random value for cos(theta) and calculate sin(theta) from there.
                    double sintheta = sqrt(1 - costheta*costheta);
                    kickDir[0] += (sintheta * cos(phi));
                    kickDir[1] += (sintheta * sin(phi));
                    kickDir[2] += costheta;

                }
                else{
                    //stimulated emission, undoes the previous absorption event.
                    kickDir[coolingAxis] += -1;
                    ionLevels[a]=0;
                }

            }

        }
        upperPop += (ionLevels[a]*1.0)/numIons;
        velocityList[a][0] += 0.001* hbar * wavevector * kickDir[0] /camass; //SI->omm * momentum * direction / mass = an omm velocity
        velocityList[a][1] += 0.001* hbar * wavevector * kickDir[1] /camass;
        velocityList[a][2] += 0.001* hbar * wavevector * kickDir[2] /camass;




        //background heating kick

        if(drand48()<bgKickProb){
            double phi  = drand48()*2*localPi; //map a random number from [0,1] -> [0,2Pi] for phi, [0,Pi] for theta
            double costheta = 2*drand48() - 1; //turns out it's actually quicker + more random to generate a random value for cos(theta) and calculate sin(theta) from there.
            double sintheta = sqrt(1 - costheta*costheta);
            xkick = (sintheta * cos(phi));
            ykick = (sintheta * sin(phi));
            zkick = costheta;

            velocityList[a][0] += 0.001* bgKickSize * xkick ; //SI->omm * momentum * direction / mass = an omm velocity
            velocityList[a][1] += 0.001* bgKickSize * ykick ;
            velocityList[a][2] += 0.001* bgKickSize * zkick ;
        }


     //constant heating term (copied from background heating kick)

         if(drand48()<heatingKickProb){
            double phi  = drand48()*2*localPi;
            double costheta = 2*drand48() - 1;
            double sintheta = sqrt(1 - costheta*costheta);
            xkick = (sintheta * cos(phi));
            ykick = (sintheta * sin(phi));
            zkick = costheta;

            heatingKickSize = heating*1e-7;
            velocityList[a][0] += heatingKickSize * xkick;
            velocityList[a][1] += heatingKickSize * ykick;
            velocityList[a][2] += heatingKickSize * zkick;
        }


        if(doFreeze==1){
            velocityList[a][0]=0.0;
            velocityList[a][1]=0.0;
            velocityList[a][2]=0.0;
        }


    }

    if(secularTempWorking*1000 > 200 && stepNum > 2000){
        freezeWaiting = 2;
    }


    if(isSecularStep==1){
        secularTemp = secularTempWorking;

        //appends the current secular temperature to the averaging list, then pops one off if needed to keep the averaging range bounded.
        secularTempAverage.push_back(secularTempWorking);
        if(secularTempAverage.size() > secularAverageRange){
            secularTempAverage.pop_front();
        }



        if(isControlTemp==true && secularTempWorking*1000 <= 1000 && stepNum > 10000){
          correction = update_pid(secularTempWorking*1000, 1);
          for (int a=0; a<numIons; ++a){
              double tmp0 = velocityList[a][0]*velocityList[a][0] + 1e-6 * correction;
              double tmp1 = velocityList[a][1]*velocityList[a][1] + 1e-6 * correction;
              double tmp2 = velocityList[a][2]*velocityList[a][2] + 1e-6 * correction;
              if(tmp0 < 0){
                a=a;//velocityList[a][0] = 0;//1e-12 * (velocityList[a][0] > 0? 1:-1);
              }
              else{
                velocityList[a][0] = sqrt(tmp0) * (velocityList[a][0] > 0? 1:-1);
              }
              if(tmp1 < 0){
                a=a;//velocityList[a][1] = 0;//1e-12 * (velocityList[a][1] > 0? 1:-1);
              }
              else{
                velocityList[a][1] = sqrt(tmp1) * (velocityList[a][1] > 0? 1:-1);
              }
              if(tmp2 < 0){
                a=a;//velocityList[a][2] = 0;//1e-12 * (velocityList[a][2] > 0? 1:-1);
              }
              else{
                velocityList[a][2] = sqrt(tmp2) * (velocityList[a][2] > 0? 1:-1);
              }
          }
        }

    }

    //frictionForce->updateParametersInContext( *(omm->context)  ); //the irritatingly slow step

    omm->context->setVelocities(velocityList); //send back any changes that have been made to the velocity
    omm->integrator->step(1);

    //window update - do this less frequently
    if(stepNum%2000==0){
        secularAverageBufferSize = secularTempAverage.size();
        averageSecularTemp = 0;
        for(int st = 0;st<secularAverageBufferSize;st++){
            averageSecularTemp+=secularTempAverage[st];
        }

                currentT = averageSecularTemp*1000/secularAverageBufferSize;

        if(doingHist == 0){
            counterTemp++;
            avgTemp = ((counterTemp - 1) * avgTemp + currentT) / counterTemp;
            if(isTempStabilized()){
                startHist();
                ofInfo << "----------------Start histogram--------------" << endl;
                cout << "----------------Start histogram--------------" << endl;

                //Reset the calculation of average temperature when histogram starts
                counterTemp = 0;
                avgTemp = 0;
            }
        }

        if (doingHist == 1){
            if(abs(currentT-targetT)>tolReachT){
                exit(0);
            }
            counterTemp++;
            avgTemp = ((counterTemp - 1) * avgTemp + currentT) / counterTemp;
        }


        ofInfo << std::fixed << std::setprecision(3) << currentTime*1e6 << "\t"
               << std::fixed << std::setprecision(3) << currentT << "\t"
               << std::fixed << std::setprecision(3) << avgTemp << endl;

        cout << "t: " << std::fixed << std::setprecision(3) << currentTime*1e6 << " us\t"
             << "T: " << std::fixed << std::setprecision(3) << currentT << " mK\t"
             << "kp: " << std::fixed << std::setprecision(3) << kp << "\t"
             << "C: " << std::fixed << std::setprecision(3) << correction << "\t"
             << "avgT: "<< std::fixed << std::setprecision(3) << avgTemp << " mK" << endl;
    }


    //update the number of histogrammed steps and, if appropriate, end the histogram.
    if(doingHist==1){
        histStepsDone++;
        if(histStepsDone>=histStepsLength){
            printf("-----------------Histogram finished--------------\n\n");
            //histogram finished, save file out.
            //ofstream histfile("histogram.hist");
            string outputHistFileName = "numIons_" + to_string(numIons) + "_targetT_" + to_string(targetT) + "_avgT_" + to_string((int)round(avgTemp)) +"mK.hist";
            cout << "Note: Histogram file saved in " << outputHistFileName << endl;
            ofstream histfile(outputHistFileName);
            histfile << "HISTMG V1.0";
            histfile.write((char*)&histXSize,sizeof(histXSize));
            histfile.write((char*)&histYSize,sizeof(histYSize));
            histfile.write((char*)&histZSize,sizeof(histZSize));
            histfile.write((char*)hist, histXSize*histYSize*histZSize*sizeof(*hist));
            histfile.close();
            delete [] hist;
            doingHist=0;
            return true;
        }
    }

    stepNum++;
    return false;

}

//------------------initialise the simulation-----------------------
void startSim(){
    if(systemInitialized==0){
        systemInitialized = 1;
        isSimRunning = 1;
        //start defining objects
        omm = new MyOpenMMData();
        OpenMM::System&   system   = *(omm->system = new OpenMM::System());


        nonbond = new OpenMM::NonbondedForce();
        system.addForce(nonbond);


        // Load any shared libraries containing GPU implementations.
        cout << "Starting simulation..."<<endl;
        printf("- loading plugins\n");
        OpenMM::Platform::loadPluginsFromDirectory(OpenMM::Platform::getDefaultPluginsDirectory());
        printf("- adding rf potential\n");

        //Define the force. If the isLinear flag is set to 1 then this makes a linear Paul trap. Else it makes a generic one with the a and q set individually.
        //CustomExternalForce, which describes the trapping potential, is a POTENTIAL, not a force!
        //The paulTrapPrefacactor in this expression comes from the substitution tau = omega*t/2, which is used in the derivation of the Mathieu equations
        //The factor 1/2 e.g. in x*x/2 comes from integrating u = [x,y,z] to get the potential from the force expression

        if(isLinear==1){
            double qr = 4*charge*VRF/(camass * pow(2*localPi*rffreq,2)*r0*r0);
            double ar = -4*kappa*charge*VDC/(camass * pow(2*localPi*rffreq,2)*z0*z0);
            paulTrapForce = new OpenMM::CustomExternalForce("paulTrapPrefactor*((ar-2*qr*rffactor)*x*x/2 + (ar-2*qr*rffactor)*y*y/2 - 2*ar*z*z/2)");
            paulTrapForce->addGlobalParameter("rffactor",1);
            paulTrapForce->addGlobalParameter("paulTrapPrefactor",pow(rffreq*2*localPi,2)*camass/4 * 602 );
            paulTrapForce->addGlobalParameter("qr",qr);
            paulTrapForce->addGlobalParameter("ar",ar);
        }

        else{

        paulTrapForce = new OpenMM::CustomExternalForce("paulTrapPrefactor*((ax-2*qx*rffactor)*x*x/2 + (ay-2*qy*rffactor)*y*y/2 + (az-2*qz*rffactor)*z*z/2)");

        paulTrapForce->addGlobalParameter("paulTrapPrefactor",pow(rffreq*2*localPi,2)*camass/4 * 602 );
        paulTrapForce->addGlobalParameter("rffactor",1);

        paulTrapForce->addGlobalParameter("ax",ax);
        paulTrapForce->addGlobalParameter("qx",qx);

        paulTrapForce->addGlobalParameter("ay",ay);
        paulTrapForce->addGlobalParameter("qy",qy);

        paulTrapForce->addGlobalParameter("az",az);
        paulTrapForce->addGlobalParameter("qz",qz);
        }

        system.addForce(paulTrapForce);


        //add friction along a specified axis x/y/z. currently neglects radiation pressure.

        frictionForce = new OpenMM::CustomExternalForce("-1 * frictionVelocity * y");
        frictionForce->addPerParticleParameter("frictionVelocity");
        system.addForce(frictionForce);



        //make some particles
        printf("- making particles\n");
        std::vector<double> emptyVec;

        std::vector<OpenMM::Vec3> initPosInNm(numIons);
        int newFrictionIndex;
        double xAxisLength = 94e-6;
        double yAxisLength = 94e-6;
        double zAxisLength = 210e-6;
        for (int a = 0; a < numIons; ++a)
        {
            double thetaRand = acos(    (2*drand48()-1));
            double phiRand = 2*localPi*drand48();
            double rRand =  1e9 * pow(drand48(),1.0/3.0);
            double x0 = xAxisLength*rRand * sin(thetaRand) * cos(phiRand);
            double y0 = yAxisLength*rRand * sin(thetaRand) * sin(phiRand);
            double z0 = zAxisLength*rRand * cos(thetaRand);
            initPosInNm[a] = OpenMM::Vec3(x0,y0,z0);
            system.addParticle(40);
            nonbond->addParticle(1.0, 0.0, 0.0);
            paulTrapForce->addParticle(a,emptyVec);
            //newFrictionIndex=frictionForce->addParticle(a,{0});
            //frictionTermIndex.push_back(newFrictionIndex);
            ionLevels.push_back(0);

        }


        //define an integrator etc

        printf("- defining integrator\n");

        omm->integrator = new OpenMM::CustomIntegrator(timeStep);
        //adds a global velocity kick size variable for use in the integration
        kickTermIndex = omm->integrator->addGlobalVariable("velKickSize", velocityKickSize / 1000.0);
        omm->integrator->addPerDofVariable("fs", 0); //adds in a dummy variable to store the force at the start of the timestep
        omm->integrator->addComputePerDof("fs", "f"); //sets the dummy variable to the current force
        omm->integrator->addComputePerDof("x", "x+dt*v+0.5*dt*dt*f/m"); //updates the positions
        omm->integrator->addComputePerDof("v", "v+0.5*dt*(f+fs)/m"); //update the velocity - f is now the force with new positions, so we use the saved force from start of timestep too.
        //this also includes a random velocity kick
        omm->integrator->addUpdateContextState();
        printf("- creating context\n");
        omm->context = new OpenMM::Context(*omm->system, *omm->integrator);
        omm->context->setPositions(initPosInNm);

        printf( "- Using OpenMM platform %s\n",omm->context->getPlatform().getName().c_str() );
        cout <<endl;

        omm->context->setVelocitiesToTemperature(20e-3); //set the velocities to Boltzmann distribution
        omm->context->setParameter("rffactor", 1);
    }
}


int main( int argc, char *argv[])
{
    if (argc > 1){
        cout << "Given parameters:" << endl;
        numIons = stoi(argv[1]);
        targetT = stoi(argv[2]);
    }
    else{
        cout << "No arguments given, use the default vaules:" << endl;
    }
    cout << "- Number of ions: "<< numIons << endl;
    if(targetT <= 0){
        isControlTemp = false;
        cout << "- targetT <= 0, temperature control set to: false" << endl;
    }
    else{
        isControlTemp = true;
        cout << "- Target temperature: " << targetT << " mK" << endl;
    }

    cout << "- Background pressure: "<< bgPressure << " mbar" << endl;
    cout << "- Laser Power: " << laserPower << " mW" << endl;
    cout << "- Detuning: " << detuningFM << " fm" << endl;
    cout << "- Time length for histogram: " << histTimeLength << " s" << endl;
    cout << endl;


    //init the rng
    time_t startTime = time(0);
    srand48(1000);

    //output file storing information of ion cooling process
    string outputInfoFileName = "numIons_" + to_string(numIons) + "_targetT_" + to_string(targetT) +".process.info";
    cout << "Note: Ion cooling process infomation saved in " << outputInfoFileName << endl << endl;

    ofstream ofInfo(outputInfoFileName);
    ofInfo << "t(us)"<< "\tT(mK)"<< "\tavgT(mK)"<< endl;

    //start simulation
    startSim();

    while(!doStep(ofInfo)){
    }
    ofInfo.close();

  return 0;
}
