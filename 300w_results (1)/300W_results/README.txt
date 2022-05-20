300 FACES IN-THE-WILD CHALLENGE RESULTS
---------------------------------------
This code generates the results of both versions of the 300W Challenge (ICCV 2013, IMAVIS 2015) in the form of Cumulative Error Distribution (CED) curves. We provide scripts to generate the results in both Python and Matlab.

Please cite:
C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. "300 Faces In-The-Wild Challenge: Database and Results", Image and Vision Computing, 2015.


RESULTS FILES
-------------
In order to generate the plots, you need to have the following two folders on the same path as the provided scripts:
1) 300W_v1 that includes Baltrusaitis.txt, Hasan.txt, Jaiswal.txt, Milborrow.txt, Yan.txt and Zhou.txt.
2) 300W_v2 that includes Cech.txt, Deng.txt, Fan.txt, Martinez.txt and Uricar.txt.

Each txt file stores the values of the CED curves for indoor, outdoor and indoor+outdoor based on 68 and 51 landmark points, in the following format:

    300W Challenge <year> Results
    Participant: <paper authors and title>
    -----------------------------------------------------------
    Bin 68_all 68_indoor 68_outdoor 51_all 51_indoor 51_outdoor
    <float> <float> <float> <float> <float> <float> <float>


PYTHON
------
The Python script (plot_results_python.py) utilizes matplotlib, numpy and pathlib. First, change your current path to the extracted directory (<path_of_extracted_directory>/300W_results) and start Python from there. We highly recommend to use Jupyter Notebook. Then simply do:

>>> # optionally to get the matplotlib graphs inline the notebook
>>> %matplotlib inline
>>> from plot_results_python import plot_results
>>> plot_results(1) # for the results of the 1st version of the challenge
>>> plot_results(2) # for the results of the 2nd version of the challenge

For an explanation of the arguments of the provided method, please read the documetnation:

>>> plot_results??


MATLAB
------
In order to call the provided Matlab script (plot_results.m), first change your current path to the extracted directory (<path_of_extracted_directory>/300W_results) and then simply do:

>>> plot_results(1) # for the results of the 1st version of the challenge
>>> plot_results(2) # for the results of the 2nd version of the challenge

For an explanation of the arguments of the provided method, please read the documetnation:

>>> help plot_results

