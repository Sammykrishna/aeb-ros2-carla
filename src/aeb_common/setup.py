from setuptools import find_packages, setup

package_name = 'aeb_common'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='samanth-krishna',
    maintainer_email='samanthkrishna2001@gmail.com',
    description='QoS profile shared by the AEB nodes',
    license='Apache-2.0',
)
