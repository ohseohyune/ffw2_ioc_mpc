from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ffw2_ioc_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # XML 모델 파일 등록
        (os.path.join('share', package_name, 'models'), glob('ffw2_ioc_mpc/system_models/mujoco_models/*.xml')),
        # Assets (Mesh 파일 등) 등록 - 하위 폴더별로 등록 필요
        (os.path.join('share', package_name, 'models/assets/ffw_bg2'), glob('ffw2_ioc_mpc/system_models/mujoco_models/assets/ffw_bg2/*')),
        (os.path.join('share', package_name, 'models/assets/ffw_sg2'), glob('ffw2_ioc_mpc/system_models/mujoco_models/assets/ffw_sg2/*')),
        (os.path.join('share', package_name, 'models/assets/rh_p12_rn'), glob('ffw2_ioc_mpc/system_models/mujoco_models/assets/rh_p12_rn/*')),
        
        ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seohy',
    maintainer_email='ohseohyun0531@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
