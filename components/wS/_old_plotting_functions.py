# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
import numpy as np
import xarray as xr
import pandas as pd

import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt

# from components.windSpeed._helping_functions import *
exec("import %s; importlib.reload(%s); from %s import *" % tuple(['components.windSpeed._helping_functions']*3))


    def plot_analogue_pdfs(self, plot_ssts = [26.,27.,28.,29.], plot_combis = [(15,60,10),(16,60,10),(0,60,10)]):

        variables = [v for v in self._pdfs.dims if v not in ['sst','wind']]

        for sst_mod in plot_ssts:
            print(sst_mod)
            modified_tracks = self.mod_tracks_sst(sst_mod)

            space = xr.DataArray(modified_tracks[variables].copy().values, coords={'ID':range(self._tracks.shape[0]), 'variable':variables}, dims=['ID','variable'])
            spaceMean = space.mean('ID')
            spaceStd = space.std('ID')
            phaseSpace = (space - spaceMean) / spaceStd

            for combi in plot_combis:
                point = np.array(combi)
                point__ = (point - spaceMean) / spaceStd
                distance = (phaseSpace - point__) ** 2
                distance.loc[:,'weather_0'] = self._weather_dis[int(point[np.array(variables)=='weather_0'])].values
                distance = np.sum(distance.values, 1)

                nearest = np.argsort(distance)[:100]
                # print(space[nearest].mean('ID').values)

                pdfs = {
                    'mod_wind' : {'c':'k', 'sst':[sst_mod-0.05] * len(nearest[:20]), 'wind':modified_tracks.wind.iloc[nearest[:20]]},
                    'mod_wind_before' : {'c':'b', 'sst':[sst_mod-0.05] * len(nearest[:20]), 'wind':modified_tracks.wind_before.iloc[nearest[:20]]},
                    'raw_wind' : {'c':'r', 'sst':self._tracks.sst.iloc[nearest[:20]], 'wind':self._tracks.wind.iloc[nearest[:20]]},
                    'raw_wind_before' : {'c':'orange', 'sst':self._tracks.sst.iloc[nearest[:20]], 'wind':self._tracks.wind_before.iloc[nearest[:20]]},
                }
                for tra,name in zip([modified_tracks,self._tracks],['mod','raw']):
                    for wi in ['wind','wind_before']:
                        kernel = stats.gaussian_kde(tra.iloc[nearest,:][wi].values)
                        pdf = kernel(self._pdfs.wind.values)
                        pdf /= pdf.sum()
                        pdfs[name+'_'+wi]['pdf'] = pdf

                # plot results
                colors = sns.color_palette('Set2', len(self._quantiles))
                plt.close('all')
                fig, axes = plt.subplots(ncols=2, figsize=(8, 5), gridspec_kw={'width_ratios': [3, 1]})
                axes[0].axhline(y=point[np.array(variables)=='wind_before'], linewidth=2, color=pdfs['mod_wind_before']['c'])
                axes[1].axhline(y=point[np.array(variables)=='wind_before'], linewidth=2, color=pdfs['mod_wind_before']['c'])

                for i,qu in enumerate(self._quantiles):
                    axes[0].scatter(self._tracks.loc[modified_tracks['qu_wind'] == i, 'sst'], self._tracks.loc[modified_tracks['qu_wind'] == i, 'wind'], alpha=.2, color=colors[i])
                    axes[0].plot(x, self._wind_quR_params.loc[qu,'intercept'].values + self._wind_quR_params.loc[qu,'slope'].values * x, color=colors[i])

                for name,details in pdfs.items():
                    axes[0].scatter(details['sst'], details['wind'], color=details['c'], alpha=1, marker='+')
                    axes[1].plot(details['pdf'],self._pdfs.wind, color=details['c'], label=name)

                for i in nearest[:5]:
                    axes[0].plot([sst_mod,self._tracks.sst.iloc[i],self._tracks.sst.iloc[i],sst_mod], [modified_tracks.wind_before.iloc[i], self._tracks.wind_before.iloc[i], self._tracks.wind.iloc[i], modified_tracks.wind.iloc[i]], linestyle=':')

                axes[0].set_xlabel('SST', fontsize=16)
                axes[0].set_ylabel('wind speed', fontsize=16)
                axes[0].annotate('\n'.join([v+'='+str(c) for v,c in zip([var for var in self._pdfs.dims if var != 'wind'],combi)]), xy=(0,1), xycoords='axes fraction', ha='left', va='top')
                axes[1].axis('off')
                axes[1].annotate('wind_before', xy=(0,point[np.array(variables)=='wind_before']), color='gray', ha='left', va='center', backgroundcolor='w', fontsize=6)
                axes[1].set_ylim(axes[0].get_ylim())
                axes[1].legend(loc='top right', fontsize=6)
                plt.savefig(self._dir + 'wind_SST'+str(np.round(sst_mod,1))+'.'+'.'.join([v+str(c) for c,v in zip(point,variables)])+'.png')


    def resimulate_event(self, storm_id = 2017260.1231, N=100):
        '''
        atl._tracks.loc[(atl._tracks.year == 2013)].name
        atl._tracks.loc[(atl._tracks.year == 2017) & (atl._tracks.wind > 140)].sid
        storm_id = 2017277.11279
        storm_id = 2017260.1231
        storm_id = 2017242.16333
        storm_id = 2013251.13342
        '''
        obs = self._tracks.loc[self._tracks.storm == storm_id]

        plt.close('all')
        simu = np.zeros([N, obs.shape[0]])
        for i in range(N):
            simu[i, 0] = 30
            for d in range(1,obs.shape[0]):
                wind_before = simu[i,d-1]
                if d > 1:
                    wind_change_before = simu[i,d-1] - simu[i,d-2]
                else:
                    wind_change_before = 10
                conditions = {'sst':obs.sst.values[d], 'weather_0':obs.weather_0.values[d], 'wind_before':wind_before, 'wind_change_before':wind_change_before}

                new_wind = self.sample(conditions)

                simu[i,d] = new_wind
            plt.plot(simu[i,:], color='gray', linewidth=0.5, alpha=0.5)

        plt.ylim(0,160)
        plt.fill_between(range(obs.shape[0]),np.nanpercentile(simu, 13 ,axis=0), np.nanpercentile(simu, 87 ,axis=0), color='gray', alpha=0.5)
        plt.plot(np.median(simu,axis=0), color='gray', linewidth=2, alpha=1)
        plt.plot(np.mean(simu,axis=0), color='gray', linewidth=2, linestyle='--', alpha=1)
        plt.plot(obs.wind.values)
        plt.savefig(self._dir + 'storm_%s.png' % (storm_id))

        print(storm_id)
        print(obs.ACE.sum() / np.float(obs.shape[0]))
        print(np.sum(simu ** 2 / 1000, axis=1).mean() / np.float(obs.shape[0]))

    def resimulate_synth(self, synth_dict=None, N=100):
        if synth_dict is None:
            synth_dict = {
                'fav_warm' : {'sst':28.6, 'weather_0':15},
                'fav_cold' : {'sst':27.4, 'weather_0':15},
                'hamp_warm' : {'sst':28.6, 'weather_0':0},
                'hamp_cold' : {'sst':27.4, 'weather_0':0},
            }

        plt.close('all')
        for name,style in synth_dict.items():
            simu = np.zeros([N, 15])
            for i in range(N):
                simu[i, 0] = 30
                for d in range(1,15):
                    wind_before = simu[i,d-1]
                    if d > 1:
                        wind_change_before = simu[i,d-1] - simu[i,d-2]
                    else:
                        wind_change_before = 10
                    conditions = {'sst':style['sst'], 'weather_0':style['weather_0'], 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                    simu[i,d] = self.sample(conditions)

            plt.plot(np.median(simu,axis=0), label=name)
        plt.legend()
        plt.savefig(self._dir + 'storm_synth.png')


    def plot_pdfs(self):
        plt.close('all')
        fig,axes = plt.subplots(nrows=3, ncols=5, figsize=(12,10))
        for r,wind_change_before in zip([0,1,2], [-15,0,15]):
            for c,wind_before in zip(range(5), [30, 50, 70, 90, 110]):
                for weather_0,lsty in zip([15,0], ['-','--']):
                    for sst,color in zip([28.4,28.6,28.8,29], ['c','orange','red','m']):
                        conditions = {'sst':sst, 'weather_0':weather_0, 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                        pdf = self._pdfs.sel(conditions, method='nearest')
                        axes[r,c].axvline(wind_before, color='k')
                        axes[r,c].plot(self._pdfs.wind, pdf, color=color, linestyle=lsty)
                        # x = pdf.wind.values[np.where(0.5 < np.cumsum(pdf.values))[0][0]]
                        # axes[r,c].axvline(x, color=color, linestyle=lsty)
        plt.tight_layout()
        plt.savefig(self._dir + 'pdfs.png')


    def plot_pdfs_sst(self):
        plt.close('all')
        fig,axes = plt.subplots(nrows=3, ncols=5, figsize=(12,10))

        for r,wind_change_before in zip([0,1,2], [-15,0,15]):
            for c,wind_before in zip(range(5), [30, 50, 70, 90, 110]):
                for sst,color in zip(np.linspace(27,29,5), sns.color_palette("plasma", 5)):
                    conditions = {'sst':sst, 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                    pdf = self._pdfs.sel(conditions, method='nearest').mean('weather_0').mean('storm_day')
                    axes[r,c].axvline(wind_before, color='k')
                    axes[r,c].plot(self._pdfs.wind, pdf, color=color)
                    axes[r,c].axvline(self._pdfs.wind[np.cumsum(pdf).values >= 0.5][0], color=color)

                    # x = pdf.wind.values[np.where(0.5 < np.cumsum(pdf.values))[0][0]]
                    # axes[r,c].axvline(x, color=color, linestyle=lsty)
        plt.tight_layout()
        plt.savefig(self._dir + 'pdfs_sst.png')


'''
    wind_obj.load_pdfs()
    wind_obj.resimulate_synth()

    wind_obj.plot_pdfs()

    for storm_id in [1997246.11316,2016230.12328, 2017260.1231, 2017242.16333, 2013251.13342]:
        wind_obj.resimulate_event(storm_id)
'''



#
