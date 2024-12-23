// This Code compute some vegetation and radar indices for Tree Species Detection
//////2018
//Warm  : '2018-05-20', '2018-09-20'
//Cold :  '2018-12-01', '2019-03-01'
//////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sentinel 1

Map.addLayer(ROI1)
var SA=ROI1.first();
var SA_Fe=ee.Feature(SA);
var ROI=SA_Fe.buffer(50).bounds().geometry(); 
Map.addLayer(ROI)
 function subset(image) {
    return image.clip(ROI);
  }
  
  function MedianVV(image){
    return image.addBands(image.focal_median({radius: 5,
                                              kernelType: 'square',
                                              units: 'pixels'})
                               .rename(['Sigma0_VV']));
  }
  function MedianVH(image){
    return image.addBands(image.focal_median({radius: 5,
                                              kernelType: 'square',
                                              units: 'pixels'})
                               .rename(['Sigma0_VH']));
  }
  function maskVV(image){
    var quantil90 = image.reduceRegion({reducer:
                            ee.Reducer.percentile({percentiles: ee.List([95]),
                                                   outputNames: ee.List(['p95'])}),
                                        scale: 10,
                                        bestEffort: true});
    var Outliers = image.select('VV').lte(ee.Number(quantil90.get('VV')));
    return image.updateMask(Outliers);}
  function maskVH(image){
    var quantil90 = image.reduceRegion({reducer:
                            ee.Reducer.percentile({percentiles: ee.List([95]),
                                                   outputNames: ee.List(['p95'])}),
                                        scale: 10,
                                        bestEffort: true});
    var Outliers = image.select('VH').lte(ee.Number(quantil90.get('VH')));
    return image.updateMask(Outliers);}




var start = ee.Date('2019-05-20');
var finish = ee.Date('2019-09-20');


  var sentinel1VV = ee.ImageCollection("COPERNICUS/S1_GRD")
                      .filter(ee.Filter.eq('instrumentMode', 'IW'))
                      .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                      //.filter(ee.Filter.eq('relativeOrbitNumber_start', 82)) //Filter necessary to basin 779926
                      .filter(ee.Filter.eq('resolution_meters', 10))
                      .filterBounds(ROI) //roi: Necessary to Doce River Basin, to avoid select excess images
                      .filterDate(start, finish)
                      .select('VV')
                      .map(maskVV)
                      .map(MedianVV)
                      .select('Sigma0_VV');
 

  var sentinel1VH = ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filter(ee.Filter.eq('instrumentMode', 'IW'))
                      .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                      //.filter(ee.Filter.eq('relativeOrbitNumber_start', 82)) //Filter necessary to basin 779926
                      .filter(ee.Filter.eq('resolution_meters', 10))
                      .filterBounds(ROI) //roi: Necessary to Doce River Basin, to avoid select excess images
                      .filterDate(start, finish)
                      .select('VH')
                      .map(maskVH)
                      .map(MedianVH)
                      .select('Sigma0_VH');
  var sentinel1 = sentinel1VV.combine(sentinel1VH);
  
  function indices(image){
    var max = image.reduceRegion({reducer: ee.Reducer.max(), scale: 10,
                                  geometry: ROI, bestEffort: true});
    var DPSVIoriginal = image.expression(
      '(((VVmax - VV)+VH)/1.414213562) * ((VV+VH)/VV) * VH', {
        'VH': image.select('Sigma0_VH'),
        'VV': image.select('Sigma0_VV'),
        'VVmax': ee.Number(max.get('Sigma0_VV'))
      });
    var DPSVIm = image.expression(
      '(VV*VV+VV*VH)/1.414213562',{
        'VH': image.select('Sigma0_VH'),
        'VV': image.select('Sigma0_VV')
      });
    
    return image.addBands([DPSVIm.rename("DPSVIm"),
                           DPSVIoriginal.rename("DPSVIo")]);}
  function DPSVInorm(image){
    var max = image.reduceRegion({reducer: ee.Reducer.max(),
                                  scale: 10,
                                  geometry: ROI,
                                  bestEffort: true});
    var min = image.reduceRegion({reducer: ee.Reducer.min(),
                                  scale: 10,
                                  geometry: ROI,
                                  bestEffort: true});
    var DPSVI = image.expression(
      '(DPSVI - DPSVImin) /(DPSVImax - DPSVImin)',{
        'DPSVI': image.select('DPSVIm'),
        'DPSVImax': ee.Number(max.get('DPSVIm')),
        'DPSVImin': ee.Number(min.get('DPSVIm'))
      });
    return image.addBands(DPSVI.rename('DPSVI'));
  }
  function CrossRation(image){
      var CR = image.expression(
      'VH-VV',{
        'VH': image.select('Sigma0_VH'),
        'VV': image.select('Sigma0_VV'),
      });
    return image.addBands(CR.rename('CR'));
  }
  var S1 = sentinel1.map(indices);
  var S1new0=S1.map(CrossRation);
  var FinalS1 = S1new0.map(DPSVInorm)
                .select(['Sigma0_VV', 'Sigma0_VH', 'DPSVIo', 'DPSVI','CR'])
                .map(subset).mean();
print('Final DS Sentinel 1',FinalS1);
//////////////////////////////////////////////////////////////////////////
