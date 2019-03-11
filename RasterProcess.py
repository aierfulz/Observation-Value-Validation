# -*- coding : utf-8 -*-
# C:/python3.6
import os
from osgeo import gdal
import numpy as np


class RasterTool(object):
    """

    """
    def __init__(self):
        pass

    @classmethod
    def read_tiff(cls, ops, *args, **kwargs):
        """
        :param ops:
        :param args:
        :param kwargs:
        :return:
        """
        gdal.UseExceptions()
        try:
            image = gdal.Open(ops)
            col, line = image.RasterXSize, image.RasterYSize
            band = image.RasterCount
            """
            return a single band image pixel values
            """
            if band is 1:
                dims = image.GetRasterBand(1)
                pixel_values = dims.ReadAsArray(0, 0, col, line)
                return pixel_values
            else:
                """
                return a multiple band image pixel values
                """
                pixel_values = np.zeros((band, col, line))
                for s in range(band):
                    dims = image.GetRasterBand(s)
                    pixel_values[s, ::] = dims.ReadAsArray(0, 0, col, line)
                return pixel_values

        except RuntimeWarning:
            "open images failed. check file path or make sure you are opening a good images"

    @classmethod
    def get_attribute_value(cls, ops, *args, **kwargs):
        """
        :param ops:
        :param args:
        :param kwargs:
        :return:
        """
        try:
            image = gdal.Open(ops)
            projection_ref = image.GetProjectionRef()
            transform = image.GetGeoTransform()
            return transform, projection_ref
        except RuntimeWarning:
            """
            Get attribute of image failed!
            """
    @classmethod
    def create_tiff(cls, data, bands, filename, out_path, trans, projection):
        """
        :param data:
        :param bands
        :param filename:
        :param out_path:
        :param trans:
        :param projection:
        :return:
        """
        def check_data_type(da):
            """
            :param da:
            :return:
            """
            if (da.dtype == 'int32') or (da.dtype == 'int') or (da.dtype == 'int16') or (da.dtype == 'int8') \
                    or (da.dtype == 'int64'):

                return gdal.GDT_Int32

            elif (da.dtype == 'float64') or (da.dtype == 'float') or (da.dtype == 'float16') or (da.dtype == 'float32'):
                return gdal.GDT_Float64

        if bands is 1:
            """
            write a single band image 
            """
            col, lines = np.shape(data)
            d2type = check_data_type(data)

            images = gdal.GetDriverByName('GTiff')
            out2images = images.Create(os.path.join(out_path, filename + '.tif'), lines, col, bands, d2type)

            band = out2images.GetRasterBand(1)
            band.WriteArray(data)
            out2images.SetGeoTransform(trans)
            out2images.SetProjection(projection)
        else:
            """
            write a multiple bands image 
            """
            col, lines = np.shape(data[0, ::])
            d2type = check_data_type(data)

            images = gdal.GetDriverByName('GTiff')
            images.Create(os.path.join(out_path, filename), col, lines, bands, d2type)

            for s in range(bands):
                band = images.GetRasterBand(1)
                band.WriteArray(data)
                images.SetGeoTransform(trans)
                images.SetProjection(projection)
