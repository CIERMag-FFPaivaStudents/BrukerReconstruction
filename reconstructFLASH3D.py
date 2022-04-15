# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Read Bruker MRI raw data from Paravision v5.1 of FLASH 3D images and save magnitude and phase images in NIFTI format. This is a simplified version inspired on Bernd U. Foerster reconstruction (https://github.com/bfoe/BrukerOfflineReco).

"""

import os
import gc
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def readRAW(path, inputName):
    """Function to read RAW data from Paravision v5.1

    Parameters
    ----------
    inputPath: string
        Path from raw files directories.

    Returns
    -------
    rawComplexData: complex array
        Unprocessed raw data in complex notation from fid directory.

    """
    
    with open(path+inputName, 'rb') as dataFile:
        rawData = np.fromfile(dataFile, dtype=np.int32)
    
    rawComplexData = rawData[0::2] + 1j*rawData[1::2]

    return rawComplexData

def readParameters(path):
    """Function to read either basic scan parameters or base level acquisition parameters from method or acqp files.

    Parameters
    ----------
    inputPath: string
        Path from parameter files directories (method or acqp).

    Returns
    -------
    parameterDict: dict
        Parameter dictionary from method or acqp files.

    """

    parameterDict = {}

    with open(path, 'r') as parameterFile:
        while True:

            line = parameterFile.readline()

            if not line:
                break

            # '$$ /' indicates when line contains original file name
            if line.startswith('$$ /'):
                originalFileName = line[line.find('/nmr/')+5:]
                originalFileName = originalFileName[0:len(originalFileName)-8]
                originalFileName = originalFileName.replace(".", "_")
                originalFileName = originalFileName.replace("/", "_")

            # '##$' indicates when line contains parameter
            if line.startswith('##$'):
                parameterName, currentLine = line[3:].split('=')

                # checks if entry is arraysize
                if currentLine[0:2] == "( " and currentLine[-3:-1] == " )":
                    parameterValue = parseArray(parameterFile, currentLine) 

                # checks if entry is struct/list
                elif currentLine[0:2] == "( " and currentLine[-3:-1] != " )":
                    while currentLine[-2] != ")": #in case of multiple lines
                        currentLine = currentLine[0:-1] + parameterFile.readline()

                    parameterValue = [parseSingleValue(lineValue) 
                            for lineValue in currentLine[1:-2].split(', ')]

                # last option is single string or number
                else:
                    parameterValue = parseSingleValue(currentLine)

                parameterDict[parameterName] = parameterValue
               # print(parameterName)
               # print(parameterValue)

    return originalFileName, parameterDict

def parseArray(parameterFile, line):
    """Parse array type from readParameters function.

    Parameters
    ----------
    parameterFile: string
        Path from parameter files directories (method or acqp).
    line: string
        Current line from file read in readParameters funciton.

    Returns
    -------
    valueList: string, int, or float
        Parsed values from input line array.
    """

    line = line[1:-2].replace(" ", "").split(",")
    arraySize = np.array([int(arrayValue) for arrayValue in line])

    valueList = parameterFile.readline().split()
    
    # If the value cannot be converted to float then it is a string
    try:
        float(valueList[0])
    except ValueError:
        return " ".join(valueList)

    while len(valueList) != np.prod(arraySize): # in case of multiple lines
        valueList = valueList + parameterFile.readline().split()

    # If the value is not an int then it is a float
    try:
        valueList = [int(singleValue) for singleValue in valueList]
    except ValueError:
        valueList = [float(singleValue) for singleValue in valueList]

    # transform list to numpy array
    if len(valueList) > 1:
        return np.reshape(np.array(valueList), arraySize)
    else:
        return valueList[0]

def parseSingleValue(singleValue):
    """Parse single value from readParameters function.

    Parameters
    ----------
    singleValue: int, float, or string
        Single value from readParameters function.

    Returns
    -------
    singleValueParsed: int, float, or string
        Parsed value from input.

    """
    
    #if it is not int then it is a float or string
    try:
        singleValueParsed = int(singleValue)
    except ValueError:
        #if it is not a float then it is a string
        try:
            singleValueParsed = float(singleValue)
        except ValueError:
            singleValueParsed = singleValue.rstrip('\n')
    
    return singleValueParsed

def checkDataImplementation(methodData):
    """Check for unexpected and not implemented data.

    Parameters
    ----------
    methodData: dict
        Parameter dictionary from method file.

    """
    if  not(methodData["Method"] == "FLASH" or methodData["Method"] == "FISP" or methodData["Method"] =="GEFC") or methodData["PVM_SpatDimEnum"] != "3D":
        print ('ERROR: Recon only implemented for FLASH/FISP 3D method');
        sys.exit(1)
    if methodData["PVM_NSPacks"] != 1:
        print ('ERROR: Recon only implemented 1 package');
        sys.exit(1)
    if methodData["PVM_NRepetitions"] != 1:
        print ('ERROR: Recon only implemented 1 repetition');
        sys.exit(1)
    if methodData["PVM_EncPpiAccel1"] != 1 or methodData["PVM_EncNReceivers"] != 1 or\
        methodData["PVM_EncZfAccel1"] != 1 or methodData["PVM_EncZfAccel2"] != 1:
        print ('ERROR: Recon for parallel acquisition not implemented');
        sys.exit(1)

def prepareData(rawComplexData, methodData):
    """Prepare raw data with a series of functions.

    """

    dim = methodData["PVM_EncMatrix"]
    EncPftAccel1 = methodData["PVM_EncPftAccel1"] 
    EncSteps1 = methodData["PVM_EncSteps1"]
    EncSteps2 = methodData["PVM_EncSteps2"]
    SPackArrPhase1Offset = methodData["PVM_SPackArrPhase1Offset"]
    SPackArrSliceOffset = methodData["PVM_SPackArrSliceOffset"]
    Fov = methodData["PVM_Fov"]
    AntiAlias = methodData["PVM_AntiAlias"]
    SpatResol = methodData["PVM_SpatResol"]

    reshapedData = reshapeData(rawComplexData, dim)

    if EncPftAccel1 != 1:
        zerosData, dim = addZerosPartialPhaseAcq(reshapedData, EncPftAccel1, dim)
    else:
        zerosData = reshapedData
    
    reorderedData = reorderData(zerosData, dim, EncSteps1, EncSteps2)
    offsetData = applyFOVoffset(reorderedData,SPackArrPhase1Offset, SPackArrSliceOffset,Fov,AntiAlias)
    zeroFillData, dim, SpatResol, zero_fill = applyZeroFill(offsetData, dim, SpatResol)
    #rollPartialEcho()
    hanningData = applyHanningFilter(zeroFillData, dim, zero_fill)

    preparedComplexData = hanningData
    return preparedComplexData, SpatResol

def reshapeData(rawComplexData, dim):
    """Reshape raw complex data to dimensions from method data.

    Parameters
    ----------
    rawComplexData: complex array
        Unprocessed raw data in complex notation from fid directory.
    dim: array
        Dimensions from methods data.

    Returns
    -------
    reshapedData: complex array
        Reshaped array with method dimensions.
    """

    dim0 = dim[0]
    dim0_mod_128 = dim0%120

    if dim0_mod_128 != 0: #Bruker sets readout point to a multiple of 128
        dim0 = (int(dim0/128+1))*128

    try: # order="F" parameter for Fortran style order as by Bruker conventions
        rawComplexData = rawComplexData.reshape(dim0, dim[1], dim[2], order='F')
    except:
        print('ERROR: k-space data reshape failed (dimension problem)')
        sys.exit(1)

    if dim0 != dim[0]:
        reshapedData = rawComplexData[0:dim[0], :, :]
    else:
        reshapedData = rawComplexData

    return reshapedData

def addZerosPartialPhaseAcq(reshapedData, EncPftAccel1, dim):
    """
    """
    zeros_ = np.zeros(shape=(dim[0],int(dim[1]*(float(EncPftAccel)-1.)),dim[2]))
    zerosData = np.append(reshapedData, zeros_, axis=1)
    dim = zerosData.shape
    return zerosData, dim

def reorderData(zerosData, dim, EncSteps1, EncSteps2):
    """
    """

    FIDdata_tmp=np.empty(shape=(dim[0],dim[1],dim[2]),dtype=np.complex64)
    reorderedData=np.empty(shape=(dim[0],dim[1],dim[2]),dtype=np.complex64)

    orderEncDir1= EncSteps1+dim[1]/2
    for i in range(0,orderEncDir1.shape[0]): 
        FIDdata_tmp[:,int(orderEncDir1[i]),:]=zerosData[:,i,:]
    
    orderEncDir2=EncSteps2+dim[2]/2
    for i in range(0,orderEncDir2.shape[0]): 
        reorderedData[:,:,int(orderEncDir2[i])]=FIDdata_tmp[:,:,i]

    return reorderedData

def applyFOVoffset(reorderedData, SPackArrPhase1Offset, SPackArrSliceOffset,Fov,AntiAlias):
    """
    """
    realFOV = Fov*AntiAlias

    phase_step1 = +2.*np.pi*float(SPackArrPhase1Offset)/float(realFOV[1])
    phase_step2 = -2.*np.pi*float(SPackArrSliceOffset)/float(realFOV[2])

    mag = np.abs(reorderedData[:,:,:])
    ph = np.angle(reorderedData[:,:,:])

    for i in range(0,reorderedData.shape[1]): 
        ph[:,i,:] -= float(i-int(reorderedData.shape[1]/2))*phase_step1
    for j in range(0,reorderedData.shape[2]): 
        ph[:,:,j] -= float(j-int(reorderedData.shape[2]/2))*phase_step2

    offsetData = mag * np.exp(1j*ph)

    return offsetData

def applyZeroFill(offsetData, dim, SpatResol):
    """
    """
    zero_fill=2
    SpatResol=SpatResol/zero_fill


    dim0Padding=int(dim[0]/2)
    dim1Padding=int(dim[1]/2)
    dim2Padding=int(dim[2]/2)


    zeroFillData = np.pad(offsetData, [(dim0Padding,dim0Padding), (dim1Padding,dim1Padding), (dim2Padding,dim2Padding)], mode='constant')
    dim=zeroFillData.shape

    return zeroFillData, dim, SpatResol, zero_fill

def applyHanningFilter(data, dim, zero_fill):
    """
    """
    percentage = 10

    nz = np.asarray(np.nonzero(data))
    first_x=np.amin(nz[0,:]); last_x=np.amax(nz[0,:])
    first_y=np.amin(nz[1,:]); last_y=np.amax(nz[1,:])
    first_z=np.amin(nz[2,:]); last_z=np.amax(nz[2,:])

    npoints_x = int(float(dim[0]/zero_fill)*percentage/100.)
    npoints_y = int(float(dim[1]/zero_fill)*percentage/100.)
    npoints_z = int(float(dim[2]/zero_fill)*percentage/100.)

    hanning_x = np.zeros(shape=(dim[0]),dtype=np.float32)
    hanning_y = np.zeros(shape=(dim[1]),dtype=np.float32)
    hanning_z = np.zeros(shape=(dim[2]),dtype=np.float32)

    x_ = np.linspace (1./(npoints_x-1.)*np.pi/2.,(1.-1./(npoints_x-1))*np.pi/2.,num=npoints_x)
    hanning_x [first_x:first_x+npoints_x] = np.power(np.sin(x_),2)
    hanning_x [first_x+npoints_x:last_x-npoints_x+1] = 1
    x_ = x_[::-1]
    hanning_x[last_x-npoints_x+1:last_x+1] = np.power(np.sin(x_),2)

    y_ = np.linspace (1./(npoints_y-1.)*np.pi/2.,(1.-1./(npoints_y-1))*np.pi/2.,num=npoints_y)
    hanning_y [first_y:first_y+npoints_y] = np.power(np.sin(y_),2)
    hanning_y [first_y+npoints_y:last_y-npoints_y+1] = 1
    y_ = y_[::-1]
    hanning_y[last_y-npoints_y+1:last_y+1] = np.power(np.sin(y_),2)

    z_ = np.linspace (1./(npoints_z-1.)*np.pi/2.,(1.-1./(npoints_z-1))*np.pi/2.,num=npoints_z)
    hanning_z [first_z:first_z+npoints_z] = np.power(np.sin(z_),2)
    hanning_z [first_z+npoints_z:last_z-npoints_z+1] = 1
    z_ = z_[::-1]
    hanning_z[last_z-npoints_z+1:last_z+1] = np.power(np.sin(z_),2)

    hanningData = data
    hanningData *= hanning_x[:, None, None]
    hanningData *= hanning_y[None, :, None]
    hanningData *= hanning_z[None, None, :]
    return hanningData

def applyFFT(data):
    """
    """

    shiftedData = np.fft.fftshift(data, axes=(0,1,2))
    transfData = shiftedData
    for k in range(0,data.shape[1]):
        transfData[:,k,:] = np.fft.fft(shiftedData[:,k,:], axis=(0))
    for i in range(0,data.shape[0]):
        transfData[i,:,:] = np.fft.fft(transfData[i,:,:], axis=(0))
    for i in range(0,data.shape[0]):
        transfData[i,:,:] = np.fft.fft(transfData[i,:,:], axis=(1))
    transfData = np.fft.fftshift(transfData, axes=(0,1,2))
    return transfData

def calculateMagnitude(spatialDomainData, acqpData, methodData):
    """
    """
    ReceiverGain = acqpData["RG"] # RG is a simple attenuation FACTOR, NOT in dezibel (dB) unit 
    n_Averages = methodData["PVM_NAverages"]

    magnitudeData = np.abs(spatialDomainData)/RG_to_voltage(ReceiverGain)/n_Averages; 
    max_ABS = np.amax(magnitudeData);
    magnitudeData *= 32767./max_ABS
    magnitudeData = magnitudeData.astype(np.int16)

    return magnitudeData

def calculatePhase(spatialDomainData):
    """
    """

    phaseData = np.angle(spatialDomainData)

    return phaseData 

def RG_to_voltage(RG):
    """
    """

    return np.power(10,11.995/20.) * np.power(RG,19.936/20.)

def saveNIFTI(path, outputName, originalFileName, suffix, data, SpatResol):
    """
    """
    affineMatrix = np.eye(4)

    affineMatrix[0,0] = SpatResol[0]*1000
    affineMatrix[0,3] = -(data.shape[0]/2)*affineMatrix[0,0]
    affineMatrix[1,1] = SpatResol[1]*1000
    affineMatrix[1,3] = -(data.shape[1]/2)*affineMatrix[1,1]
    affineMatrix[2,2] = SpatResol[2]*1000
    affineMatrix[2,3] = -(data.shape[2]/2)*affineMatrix[2,2]

    NIFTIimg = nib.Nifti1Image(data, affineMatrix)
    NIFTIimg.header.set_slope_inter(np.amax(data)/32767.,0)
    NIFTIimg.header.set_xyzt_units(3, 8)
    NIFTIimg.set_sform(affineMatrix, code=0)
    NIFTIimg.set_qform(affineMatrix, code=1)

    nib.save(NIFTIimg, path+outputName+originalFileName+suffix+'.nii.gz')


if __name__ == '__main__':

    inputPath = '/home/solcia/Documents/phd/MRI data/Coral/6'
    outputPath = '/home/solcia/Documents/phd/MRI data/Coral'

    inputName = '/ser'
    outputName = '/FLASH3D_'

    _,acqpData = readParameters(inputPath+'/acqp') #acqp stands for acquisition parameters
    originalFileName, methodData = readParameters(inputPath+'/method') # methods contains basic scan parameters

    checkDataImplementation(methodData)

    rawComplexData = readRAW(inputPath, inputName)
    
    preparedComplexData, SpatResol = prepareData(rawComplexData, methodData)
   
    img_slice = 80

    plt.figure()
    plt.imshow(np.absolute(preparedComplexData[:,:,img_slice]), cmap='gray')

    spatialDomainData = applyFFT(preparedComplexData)

    magnitudeData = calculateMagnitude(spatialDomainData, acqpData, methodData)
    
#    plt.figure()
#    plt.imshow(magnitudeData[:,:,img_slice], cmap='gray')

    phaseData = calculatePhase(spatialDomainData)

#    plt.figure()
#    plt.imshow(phaseData[:,:,img_slice], cmap='gray')
#    plt.show()

    saveNIFTI(outputPath, outputName, originalFileName, '_MAGNT', magnitudeData, SpatResol)
    saveNIFTI(outputPath, outputName, originalFileName, '_PHASE', phaseData, SpatResol)
