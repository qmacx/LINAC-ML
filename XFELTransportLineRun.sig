SDDS1
!# little-endian
&description text="sigma matrix--input: XFELTransportLineRun.ele  lattice: XFELTransportLineFinal.lte", contents="sigma matrix", &end
&parameter name=Step, description="Simulation step", type=long, &end
&parameter name=SVNVersion, description="SVN version number", type=string, fixed_value=unknown, &end
&column name=s, units=m, description=Distance, type=double,  &end
&column name=ElementName, description="Element name", format_string=%10s, type=string,  &end
&column name=ElementOccurence, description="Occurence of element", format_string=%6ld, type=long,  &end
&column name=ElementType, description="Element-type name", format_string=%10s, type=string,  &end
&column name=s1, symbol="$gs$r$b1$n", units=m, description="sqrt(<x*x>)", type=double,  &end
&column name=s12, symbol="$gs$r$b12$n", units=m, description="<x*xp'>", type=double,  &end
&column name=s13, symbol="$gs$r$b13$n", units="m$a2$n", description="<x*y>", type=double,  &end
&column name=s14, symbol="$gs$r$b14$n", units=m, description="<x*y'>", type=double,  &end
&column name=s15, symbol="$gs$r$b15$n", units="m$a2$n", description="<x*s>", type=double,  &end
&column name=s16, symbol="$gs$r$b16$n", units=m, description="<x*delta>", type=double,  &end
&column name=s17, symbol="$gs$r$b17$n", units="m*s", description="<x*t>", type=double,  &end
&column name=s2, symbol="$gs$r$b2$n", description="sqrt(<x'*x'>)", type=double,  &end
&column name=s23, symbol="$gs$r$b23$n", units=m, description="<x'*y>", type=double,  &end
&column name=s24, symbol="$gs$r$b24$n", description="<x'*y'>", type=double,  &end
&column name=s25, symbol="$gs$r$b25$n", units=m, description="<x'*s>", type=double,  &end
&column name=s26, symbol="$gs$r$b26$n", description="<x'*delta>", type=double,  &end
&column name=s27, symbol="$gs$r$b27$n", units=s, description="<x'*t>", type=double,  &end
&column name=s3, symbol="$gs$r$b3$n", units=m, description="sqrt(<y*y>)", type=double,  &end
&column name=s34, symbol="$gs$r$b34$n", units=m, description="<y*y'>", type=double,  &end
&column name=s35, symbol="$gs$r$b35$n", units="m$a2$n", description="<y*s>", type=double,  &end
&column name=s36, symbol="$gs$r$b36$n", units=m, description="<y*delta>", type=double,  &end
&column name=s37, symbol="$gs$r$b37$n", units="m*s", description="<y*t>", type=double,  &end
&column name=s4, symbol="$gs$r$b4$n", description="sqrt(<y'*y')>", type=double,  &end
&column name=s45, symbol="$gs$r$b45$n", units=m, description="<y'*s>", type=double,  &end
&column name=s46, symbol="$gs$r$b46$n", description="<y'*delta>", type=double,  &end
&column name=s47, symbol="$gs$r$b47$n", units=s, description="<y'*t>", type=double,  &end
&column name=s5, symbol="$gs$r$b5$n", units=m, description="sqrt(<s*s>)", type=double,  &end
&column name=s56, symbol="$gs$r$b56$n", units=m, description="<s*delta>", type=double,  &end
&column name=s57, symbol="$gs$r$b57$n", units="m*s", description="<s*t>", type=double,  &end
&column name=s6, symbol="$gs$r$b6$n", description="sqrt(<delta*delta>)", type=double,  &end
&column name=s67, symbol="$gs$r$b67$n", units=s, description="<delta*t>", type=double,  &end
&column name=s7, symbol="$gs$r$b7$n", description="sqrt(<t*t>)", type=double,  &end
&column name=ma1, symbol="max$sb$ex$sb$e", units=m, description="maximum absolute value of x", type=double,  &end
&column name=ma2, symbol="max$sb$ex'$sb$e", description="maximum absolute value of x'", type=double,  &end
&column name=ma3, symbol="max$sb$ey$sb$e", units=m, description="maximum absolute value of y", type=double,  &end
&column name=ma4, symbol="max$sb$ey'$sb$e", description="maximum absolute value of y'", type=double,  &end
&column name=ma5, symbol="max$sb$e$gD$rs$sb$e", units=m, description="maximum absolute deviation of s", type=double,  &end
&column name=ma6, symbol="max$sb$e$gd$r$sb$e", description="maximum absolute value of delta", type=double,  &end
&column name=ma7, symbol="max$sb$e$gD$rt$sb$e", units=s, description="maximum absolute deviation of t", type=double,  &end
&column name=minimum1, symbol="x$bmin$n", units=m, type=double,  &end
&column name=minimum2, symbol="x'$bmin$n", units=m, type=double,  &end
&column name=minimum3, symbol="y$bmin$n", units=m, type=double,  &end
&column name=minimum4, symbol="y'$bmin$n", units=m, type=double,  &end
&column name=minimum5, symbol="$gD$rs$bmin$n", units=m, type=double,  &end
&column name=minimum6, symbol="$gd$r$bmin$n", units=m, type=double,  &end
&column name=minimum7, symbol="t$bmin$n", units=s, type=double,  &end
&column name=maximum1, symbol="x$bmax$n", units=m, type=double,  &end
&column name=maximum2, symbol="x'$bmax$n", units=m, type=double,  &end
&column name=maximum3, symbol="y$bmax$n", units=m, type=double,  &end
&column name=maximum4, symbol="y'$bmax$n", units=m, type=double,  &end
&column name=maximum5, symbol="$gD$rs$bmax$n", units=m, type=double,  &end
&column name=maximum6, symbol="$gd$r$bmax$n", units=m, type=double,  &end
&column name=maximum7, symbol="t$bmax$n", units=s, type=double,  &end
&column name=Sx, symbol="$gs$r$bx$n", units=m, description=sqrt(<(x-<x>)^2>), type=double,  &end
&column name=Sxp, symbol="$gs$r$bx'$n", description=sqrt(<(x'-<x'>)^2>), type=double,  &end
&column name=Sy, symbol="$gs$r$by$n", units=m, description=sqrt(<(y-<y>)^2>), type=double,  &end
&column name=Syp, symbol="$gs$r$by'$n", description=sqrt(<(y'-<y'>)^2>), type=double,  &end
&column name=Ss, symbol="$gs$r$bs$n", units=m, description=sqrt(<(s-<s>)^2>), type=double,  &end
&column name=Sdelta, symbol="$gs$bd$n$r", description=sqrt(<(delta-<delta>)^2>), type=double,  &end
&column name=St, symbol="$gs$r$bt$n", units=s, description=sqrt(<(t-<t>)^2>), type=double,  &end
&column name=ex, symbol="$ge$r$bx$n", units=m, description="geometric horizontal emittance", type=double,  &end
&column name=enx, symbol="$ge$r$bx,n$n", units=m, description="normalized horizontal emittance", type=double,  &end
&column name=ecx, symbol="$ge$r$bx,c$n", units=m, description="geometric horizontal emittance less dispersive contributions", type=double,  &end
&column name=ecnx, symbol="$ge$r$bx,cn$n", units=m, description="normalized horizontal emittance less dispersive contributions", type=double,  &end
&column name=ey, symbol="$ge$r$by$n", units=m, description="geometric vertical emittance", type=double,  &end
&column name=eny, symbol="$ge$r$by,n$n", units=m, description="normalized vertical emittance", type=double,  &end
&column name=ecy, symbol="$ge$r$by,c$n", units=m, description="geometric vertical emittance less dispersive contributions", type=double,  &end
&column name=ecny, symbol="$ge$r$by,cn$n", units=m, description="normalized vertical emittance less dispersive contributions", type=double,  &end
&column name=betaxBeam, symbol="$gb$r$bx,beam$n", units=m, description="betax for the beam, excluding dispersive contributions", type=double,  &end
&column name=alphaxBeam, symbol="$ga$r$bx,beam$n", description="alphax for the beam, excluding dispersive contributions", type=double,  &end
&column name=betayBeam, symbol="$gb$r$by,beam$n", units=m, description="betay for the beam, excluding dispersive contributions", type=double,  &end
&column name=alphayBeam, symbol="$ga$r$by,beam$n", description="alphay for the beam, excluding dispersive contributions", type=double,  &end
&data mode=binary, &end
                 _BEG_      MARK�k��M��>�5�.�(u��t�/��&��J��XE=�C�U�3=��� ���"X��q;�"��>�W;�5=f���f�'=�ߢ+cp'�=
ףp-
��!g���d��d:E@�>/	-�A'u��nU�+�Zd;�O5;-���i�g��>��������/���#���DA������Ơ>�����M���%����;M @+/�0?+��Nt�Sʫ�v�<�aI��6�>���b��>cQE�>˔��]�>4NdI��>�l0h3Y�?"���<�aI��6羄��b�վy-Jd���˔��]Ѿ4NdI����l0h3Y߿"�����]�B�>/m�`�>cQE�>�����u�>��{<�>!����?߿�
��>��<�k��M��>�"��>�d:E@�>g��>�����Ơ>M @+/�0?Sʫ�v�<�_:a�q�=HP��ye>�_:a�q�=HP��ye>Z��Uq�=N�$�ye>Z��Uq�=N�$�ye>`��F� @�j����?ʙo��1	@�x`X��?           C      CHARGE�k��M��>�5�.�(u��t�/��&��J��XE=�C�U�3=��� ���"X��q;�"��>�W;�5=f���f�'=�ߢ+cp'�=
ףp-
��!g���d��d:E@�>/	-�A'u��nU�+�Zd;�O5;-���i�g��>��������/���#���DA������Ơ>�����M���%����;M @+/�0?+��Nt�Sʫ�v�<�aI��6�>���b��>cQE�>˔��]�>4NdI��>�l0h3Y�?"���<�aI��6羄��b�վy-Jd���˔��]Ѿ4NdI����l0h3Y߿"�����]�B�>/m�`�>cQE�>�����u�>��{<�>!����?߿�
��>��<�k��M��>�"��>�d:E@�>g��>�����Ơ>M @+/�0?Sʫ�v�<�_:a�q�=HP��ye>�_:a�q�=HP��ye>Z��Uq�=N�$�ye>Z��Uq�=N�$�ye>`��F� @�j����?ʙo��1	@�x`X��?           OTR1      WATCH�k��M��>�5�.�(u��t�/��&��J��XE=�C�U�3=��� ���"X��q;�"��>�W;�5=f���f�'=�ߢ+cp'�=
ףp-
��!g���d��d:E@�>/	-�A'u��nU�+�Zd;�O5;-���i�g��>��������/���#���DA������Ơ>�����M���%����;M @+/�0?+��Nt�Sʫ�v�<�aI��6�>���b��>cQE�>˔��]�>4NdI��>�l0h3Y�?"���<�aI��6羄��b�վy-Jd���˔��]Ѿ4NdI����l0h3Y߿"�����]�B�>/m�`�>cQE�>�����u�>��{<�>!����?߿�
��>��<�k��M��>�"��>�d:E@�>g��>�����Ơ>M @+/�0?Sʫ�v�<�_:a�q�=HP��ye>�_:a�q�=HP��ye>Z��Uq�=N�$�ye>Z��Uq�=N�$�ye>`��F� @�j����?ʙo��1	@�x`X��?�������?   B1   	   CSRCSBEND�+�?�M�>��h���H=j�g nI0=�=�G=�ޱ7)�2=�G�|�r�=���k�p;�`PxǠ�>>x�QV:=3	A���'=�%YW((�l�^�,�=֠�\�e������>�!]2?�\��n����-����V)P="�U���j�ǫ�k���>B��%����|kd�<�/�5��@����HȠ>%h#��2�>�~���;.+��j�0?��34%q���s��<b�~wH�*?by
�u�O?���6�>��f�]�>  �Ͱĸ>���:Y�?  ��~-�<b�~wH�*�by
�u�O����6���f�]Ѿ  �Ͱĸ����:Y߿  ��~-��z�*���'�w�V�O�	�A�v�>��]*�u�>  ��)�>�ԏ�?߿  �&��<�+�?�M�>�`PxǠ�>�����>ǫ�k���>���HȠ>.+��j�0?��s��<��(��q�=U�LVt�h>�m<�r�=�`�q7ze>CǨVq�=n�RŚye>x�Vq�=/J��ye>=�0/ @��S��b?����8V@'�A��c�?ffffff�?   BDCSR1      CSRDRIFT4�oV��>��K��=�GoV�Y=?�̸M=Z>�3U�=�My���=�k��SS;�`PxǠ�>`̠'C=3	A���'=�TvJAF*��e zo-�=�e�ڶ�g�����>�O�{=�@�N1��
�5gV=���P�n�ǫ�k���>Z��Ķ���5�۟.="	���@��Ÿɠ>w~��^;i�H�ʈ�;g��z�0?��U����PO�#�<�F���S?by
�u�O?������>��f�]�>  `X�ȸ>�@vEY�?  ��G1�<�F���S�by
�u�O���������f�]Ѿ  `X�ȸ��@vEY߿  ��G1��{7[)��R�w�V�O�y��D���>��]*�u�>  �5M�>v�i��?߿  @q3��<4�oV��>�`PxǠ�>����>ǫ�k���>�Ÿɠ>g��z�0?�PO�#�<��(��q�=�.u�r�h>t�uG�q�=]3���ye>�BǨVq�=,�y`�ye>vi��Vq�=8��P�ye>0j�|@�� k}߿��QhP�	@�9!b��пffffff�?   OTR2      WATCH4�oV��>��K��=�GoV�Y=?�̸M=Z>�3U�=�My���=�k��SS;�`PxǠ�>`̠'C=3	A���'=�TvJAF*��e zo-�=�e�ڶ�g�����>�O�{=�@�N1��
�5gV=���P�n�ǫ�k���>Z��Ķ���5�۟.="	���@��Ÿɠ>w~��^;i�H�ʈ�;g��z�0?��U����PO�#�<�F���S?by
�u�O?������>��f�]�>  `X�ȸ>�@vEY�?  ��G1�<�F���S�by
�u�O���������f�]Ѿ  `X�ȸ��@vEY߿  ��G1��{7[)��R�w�V�O�y��D���>��]*�u�>  �5M�>v�i��?߿  @q3��<4�oV��>�`PxǠ�>����>ǫ�k���>�Ÿɠ>g��z�0?�PO�#�<��(��q�=�.u�r�h>t�uG�q�=]3���ye>�BǨVq�=,�y`�ye>vi��Vq�=8��P�ye>0j�|@�� k}߿��QhP�	@�9!b��п�������?   B2   	   CSRCSBENDn�C��>̮8}=�=)�����a=7����P=j�m}*�B|�l�=���%��R��H&��>�&�a!ME=�����'=��XQ�+�`�"�C��>���3h�Aj�<�>����zۄ=���Z	c2�P�|W�3g=>�V�vp���x�>>a���A�R�5���4=p۷%F>A���2�*ɠ>�Q�j�s��Z��;����0?�ab��˱�Z��lh�<)[�>�-V?$�̰ ��>�����>���"�]�>   h��>�"c�FY�?   e�"�<)[�>�-V�$�̰ �վ��������"�]Ѿ   h����"c�FY߿   e�"��\�X�U��0�jd]�>AN��54�>�@V��u�>  �߮�>��f�?߿   4��<n�C��>�H&��>Aj�<�>��x�>��2�*ɠ>����0?Z��lh�<�k�h�=yf���g>�R��1r�=p@6Jze>�߮�Uq�=���ӗye>Ըw�Uq�=����ye>C{G%�@?���%*cGE�@�90�=ٿ�������?   BDCSR2      CSRDRIFTL$-?�~�>�\��ڔ=�\����d=�ݧ���P=l�ݤT��`F�U��=�R��\��H&��>p���C}F=�����'=_���+�
�I4�g>��2}u�3h��w�ܳ�>�����Y�=^3ؓJ�2��J�Mg=��Y�p���x�>�����A����$t�:=���M>A����*ɠ>�Lu[�u�<I��Z��;=�"e>�0?�n�����h�<�9[��.V?$�̰ ��>�v3D��>���"�]�>  �|h��>xzzGY�?   }�"�<�9[��.V�$�̰ �վ�v3D�����"�]Ѿ  �|h���xzzGY߿   }�"��c����U��0�jd]�>
,��j`�>�@V��u�>  `���>vo�6�?߿  �����<L$-?�~�>�H&��>�w�ܳ�>��x�>���*ɠ>=�"e>�0?���h�<Ӭk�h�=�C�F�g>�4�r�=|6@{8ze>�߮�Uq�=Sa��ye>
�ܙUq�=5�LI�ye>t�	_�@
@�J��<�Q���g\@ @v\�wݿ8��m4  @   RCOLB2      RCOL)�J�~�>2�w۔=�	D��d=���/ȥP=���������I��=co.K�\��H&��>b��j}F=�����'=�����+�
�I4�g>�ޗ>u�3h�'�����>2fZ�=8U�ZZ�2��l��,Mg=hY�y�p���x�>����A����$t�:=|��M>A�����*ɠ>F��Z�u�1 ��Z��;=�"e>�0?��ia��Z�×h�<v���.V?$�̰ ��>��p���>���"�]�>  �|h��>xzzGY�?  @}�"�<v���.V�$�̰ �վ��p������"�]Ѿ  �|h���xzzGY߿  @}�"��=�y��U��0�jd]�>�p�&�`�>�@V��u�>  @���>vo�6�?߿  ੀ��<)�J�~�>�H&��>'�����>��x�>����*ɠ>=�"e>�0?Z�×h�<�k�h�=�C�F�g>�4�r�=�6@{8ze>�߮�Uq�=Sa��ye>
�ܙUq�=5�LI�ye>���@
@��K�9=����\@2��Jxݿ8��m4  @   OTR3      WATCH)�J�~�>2�w۔=�	D��d=���/ȥP=���������I��=co.K�\��H&��>b��j}F=�����'=�����+�
�I4�g>�ޗ>u�3h�'�����>2fZ�=8U�ZZ�2��l��,Mg=hY�y�p���x�>����A����$t�:=|��M>A�����*ɠ>F��Z�u�1 ��Z��;=�"e>�0?��ia��Z�×h�<v���.V?$�̰ ��>��p���>���"�]�>  �|h��>xzzGY�?  @}�"�<v���.V�$�̰ �վ��p������"�]Ѿ  �|h���xzzGY߿  @}�"��=�y��U��0�jd]�>�p�&�`�>�@V��u�>  @���>vo�6�?߿  ੀ��<)�J�~�>�H&��>'�����>��x�>����*ɠ>=�"e>�0?Z�×h�<�k�h�=�C�F�g>�4�r�=�6@{8ze>�߮�Uq�=Sa��ye>
�ܙUq�=5�LI�ye>���@
@��K�9=����\@2��Jxݿk	��g3@   B3   	   CSRCSBEND��d!�>2��
�=��,9�fj=�����Q=Z�v��3�K��eƕ�=a�s<��q��.j�͡�>���(3kH=�{��w'=s��S*�6���2�t�'ZȎg��/���>�M�9V�=Jx�(4�y�m�2�q=eI����q��j�᣷>v�Ĭl��	8+���F=�1)p��A���� �ɠ>/���S|�������;Q��hſ0?"ȿ4y��t�T��<�J^seS?A���O?CnuG~��>a�W�]�>  ��)��>��DY�?  @#�<�J^seS�ʄ�i�*O?CnuG~��a�W�]Ѿ  ��)�����DY߿  @#�����ŗR�A���O?x����>�&y�u�>  �J���>�Tj @߿  @�k��<��d!�>�.j�͡�>�/���>�j�᣷>��� �ɠ>Q��hſ0?t�T��<��?+�G�=�d�(Qo>�uh�cr�=M���qze>��Vq�=�%�$�ye>�Q�Vq�=Ѥ"��ye>�)�@�v￥�I��]@�g���k	��g3@   BDCSR1      CSRDRIFTr/���&�>�E[u*�=�@�`xqu=�"P{�T=|���>��O�6f�=����V{��.j�͡�>�z�#IN=�{��w'=�p���4(��/'�d0�n���֧e�$��)Z�>X��Ċf�=�i�p6���d��y=j&nv�t��j�᣷>���=���j���(!R=Li�"�A�=}�Dɠ>�_��ჽك-��;B�h�0?�)ܐJ���n#ƬS�<��[Η@+?A���O?���)�>a�W�]�>  �TZ��>��eoLY�?  �"�<��[Η@+�ʄ�i�*O?���)��a�W�]Ѿ  �TZ�����eoLY߿  �"���`шE�&�A���O?�͛B�J�>�&y�u�>  �����>nb&�?߿  @ϗ�<r/���&�>�.j�͡�>$��)Z�>�j�᣷>=}�Dɠ>B�h�0?n#ƬS�<��?+�G�=����%Qo>:(�]�t�=�{��=|e>��Vq�=���ye>c4��Uq�=�t��ye>��J��@"�o����µ�"Km@��S���k	��g3@   OTR4      WATCHr/���&�>�E[u*�=�@�`xqu=�"P{�T=|���>��O�6f�=����V{��.j�͡�>�z�#IN=�{��w'=�p���4(��/'�d0�n���֧e�$��)Z�>X��Ċf�=�i�p6���d��y=j&nv�t��j�᣷>���=���j���(!R=Li�"�A�=}�Dɠ>�_��ჽك-��;B�h�0?�)ܐJ���n#ƬS�<��[Η@+?A���O?���)�>a�W�]�>  �TZ��>��eoLY�?  �"�<��[Η@+�ʄ�i�*O?���)��a�W�]Ѿ  �TZ�����eoLY߿  �"���`шE�&�A���O?�͛B�J�>�&y�u�>  �����>nb&�?߿  @ϗ�<r/���&�>�.j�͡�>$��)Z�>�j�᣷>=}�Dɠ>B�h�0?n#ƬS�<��?+�G�=����%Qo>:(�]�t�=�{��=|e>��Vq�=���ye>c4��Uq�=�t��ye>��J��@"�o����µ�"Km@��S����<,Ԛf@   B4   	   CSRCSBEND���>���E�2�=�>De,y=�� 
?�U=���E�6�)�@��h�ǐ�uemt�C]A�>����P=�B��I(=O�X�n'��~/^?Y==���H6�d��$�n�>�����=�q�e�7�yo�t�}=����]�t�$���ˣ�>�@�(���4d����R=���7A�Mh.XǠ>�C��t���E9����;������0?a�}3Ig»n�j�&�<���5G9�>������>;s��D�>����]�>  `X̽�>���LY�?  �q'�<§��w�����վ;s��D������]Ѿ  `X̽�����LY߿  �q'�����5G9�>Ntmn�^�>[�(/w�>Mv��u�>  ����>�0bc @߿  ��L��<���>C]A�>�$�n�>$���ˣ�>Mh.XǠ>������0?n�j�&�<�ik��t�=�1�ډ|e>3�^��t�=�z=Ԉ|e>-�Uq�=���ye>���Tq�=��A�ye>:�"���@_
!����PO�@� �)���<,Ԛf@   OTR5      WATCH���>���E�2�=�>De,y=�� 
?�U=���E�6�)�@��h�ǐ�uemt�C]A�>����P=�B��I(=O�X�n'��~/^?Y==���H6�d��$�n�>�����=�q�e�7�yo�t�}=����]�t�$���ˣ�>�@�(���4d����R=���7A�Mh.XǠ>�C��t���E9����;������0?a�}3Ig»n�j�&�<���5G9�>������>;s��D�>����]�>  `X̽�>���LY�?  �q'�<§��w�����վ;s��D������]Ѿ  `X̽�����LY߿  �q'�����5G9�>Ntmn�^�>[�(/w�>Mv��u�>  ����>�0bc @߿  ��L��<���>C]A�>�$�n�>$���ˣ�>Mh.XǠ>������0?n�j�&�<�ik��t�=�1�ډ|e>3�^��t�=�z=Ԉ|e>-�Uq�=���ye>���Tq�=��A�ye>:�"���@_
!����PO�@� �)����r�:ց@   DQM3B      DRIFJR��h�>� ip�@�=9J��'�=i�<%�Z=�f_1��D�1d�Y��b�+����{��C]A�>cʌq�U=�B��I(=+�X�n'��~/^?Y==�xÃ6�d��l�f̲�>v�z~Ԥ=�3��n�:�-Ħ����=�dz��w�$���ˣ�>-m�(���4d����R=:�p�:A��^g.XǠ>VIϋ�t���.����;������0?�3��yp»MsS.(�<�p1�i��>������>���İ��>����]�>  @X̽�>���LY�?  ��~'�<�p1�i��������վ���İ�������]Ѿ  @X̽�����LY߿  ��~'�����O��>Ntmn�^�>
e�u�>Mv��u�>  ����>�0bc @߿  ��Y��<JR��h�>C]A�>�l�f̲�>$���ˣ�>�^g.XǠ>������0?MsS.(�<�ik��t�=�1�ډ|e>4�^��t�=�z=Ԉ|e>(�Uq�=���ye>���Tq�=��A�ye>np�x�z,@�@EE���n��j%@Z�|/5���n	�@   QM4      QUAD��y ��>�����ֽ��Uv]�=%�}�N�=����/B�J���<\�9�N�H������!�>l����i���l\����8�e�E=���EYk=��~~'g�;����>ɋ>�.�=�j�ě@�#BH��=�=g ^[&�}��;����>nC�P�
E�2�j���=#��A^ׂ��^g.XǠ>UIϋ�t���ji9���;������0?���~9r»b�l(�<�$e�3��>��o>
 ?74�ao�>E���j?  �X̽�>���LY�?  @��'�<�$e�3�����o>
 �74�ao��NDʳY�  �X̽�����LY߿  @��'��j:tm�X�>�A�I�`�>Xoފ%�>E���j?  @���>�0bc @߿  �\\��<��y ��>����!�>����>�;����>�^g.XǠ>������0?b�l(�<`ik��t�=�1�ډ|e>ӯ^��t�=yz=Ԉ|e>��Uq�=f��ye>)��Tq�=���A�ye>{�b)��$@`�2��+@5u��&�0@��S�5��i�r��@   DQM4      DRIF"]���>޻e+������Ep��u="�Mx�Xq=�|���Y-����F�����Gj�����!�>���X%���l\����8�e�E=���EYk=���U'g�;,3?�/k�>`^����=���q�$K��3�~ �='�(�M���;����>nC�P�
E�2�j���=4���^ׂ��^g.XǠ>VIϋ�t���~V����;������0?��܌$u»ｗ�(�<� �Ha�>��o>
 ?�c��7�?E���j?  @X̽�>���LY�?  ��'�<� �Haྻ�o>
 ��c��7��NDʳY�  @X̽�����LY߿  ��'��"�7>JL�>�A�I�`�>���Ss�?E���j?  ����>�0bc @߿   �`��<"]���>����!�>,3?�/k�>�;����>�^g.XǠ>������0?ｗ�(�<�ik��t�=�1�ډ|e>G�^��t�=�z=Ԉ|e>\�Uq�=X��ye>�
��Tq�=� �A�ye>f��tX�?zgG���@+S^��F@��m$B�ל0���@   QM5      QUAD����� �>\ǡ���=��|!Z�VG��FT=~�����^�9P=�[҈̔M��hw5��>e��5�<��t^[�/�=�
f�==n�Ep=��O�/�z;#Tt�n�>��2����1���K��BPW�=�r9�8��U�ү]�>��E3�E=J1: >���t
9�`�;�^g.XǠ>VIϋ�t���u�煂�;������0?o�1�v»�J)�<]H�nk�>�����>q���?�[���?  @X̽�>���LY�?  ���'�<�iH����������q����]k����  @X̽�����LY߿  ���'��]H�nk�>��V�qi�>c�d��?�[���?  ����>�0bc @߿   c��<����� �>�hw5��>#Tt�n�>U�ү]�>�^g.XǠ>������0?�J)�<�ik��t�=�1�ډ|e>-�^��t�=�z=Ԉ|e>\�Uq�=X��ye>�
��Tq�=� �A�ye>4Z-��?Z(����ݿY���x�F@����B@�ֲ��@   DQM5      DRIFFi���0�>�O�W�=m��
�#��É+�3��=��U%=��#�WPh=�b�Dc;�hw5��>�M
��K��t^[�/�=�
f�==n�Ep=5�')/�z;��%�2�>�l�@⽌
V�F<@���/��={�n'}�U�ү]�>��E3�E=J1: >���0��`�;�^g.XǠ>VIϋ�t���aAT���;������0?�����y»7�|)�<H"uMy�>�����>S,��G��>�[���?  @X̽�>���LY�?  @�'�<H"uMy��������{^6���]k����  @X̽�����LY߿  @�'��	7�S�>��V�qi�>S,��G��>�[���?  ����>�0bc @߿   &g��<Fi���0�>�hw5��>��%�2�>U�ү]�>�^g.XǠ>������0?7�|)�<�ik��t�=�1�ډ|e>��^��t�=�z=Ԉ|e>��Uq�=`��ye>_��Tq�=���A�ye>:�8�E�@�t��:���j��0@�~����5@,
��@   QM6      QUAD���D�:�>}z7��= B{��ŀ�Ɗ�w4�^_!D�0=��(|1l=ӘC�~�m;_�,9���>�9��������p�<�`��ro!=g�.�N��~U39_;�~	��>�>�� �,^5=��f�9�3T��4��=�=R�/=w�X`���>�d�Z"�<5��^2�� �w+9;�^g.XǠ>VIϋ�t��������;������0?�π�{»S��)�<sy.#��>�;��5��>(�^@� �>�Ǐ+���>  @X̽�>���LY�?  ���'�<sy.#���,.���ǾD������Ǐ+��þ  @X̽�����LY߿  ���'���I�t�>�;��5��>(�^@� �>,f�')��>  ����>�0bc @߿  @�i��<���D�:�>_�,9���>�~	��>�>X`���>�^g.XǠ>������0?S��)�<�ik��t�=�1�ډ|e>-�^��t�=�z=Ԉ|e>,�Uq�=���ye>���Tq�=��A�ye>+ @1V�� |ۿ~�,�~&@��+�ۉ�ʉ|4
1@   DQM6      DRIFUsx�F6�>fR�(e��=��z�倽Hz��G�l��Y-5=,1�k�7h=�z�H�r;_�,9���>c{¹�������p�<�`��ro!=g�.�N��*��39_;�0��&F�>�@o5��X=�i���	9���q�W�=���}kv�X`���>�d�Z"�<5��^2���	x+9;�^g.XǠ>VIϋ�t��������;������0?��wy�~»��(*�<�}�^�>�>�;��5��>5�J��>�Ǐ+���>  @X̽�>���LY�?  �[�'�<�}�^�>���,.���Ǿ��{I'`���Ǐ+��þ  @X̽�����LY߿  �[�'���L�=q�>�;��5��>5�J��>,f�')��>  ����>�0bc @߿  ��m��<Usx�F6�>_�,9���>�0��&F�>X`���>�^g.XǠ>������0?��(*�<�ik��t�=�1�ډ|e>-�^��t�=�z=Ԉ|e>-�Uq�=���ye>���Tq�=��A�ye>�^LH�@�k
�8�࿶4_��&@����0,��ʉ|4
1@   OTR6      WATCHUsx�F6�>fR�(e��=��z�倽Hz��G�l��Y-5=,1�k�7h=�z�H�r;_�,9���>c{¹�������p�<�`��ro!=g�.�N��*��39_;�0��&F�>�@o5��X=�i���	9���q�W�=���}kv�X`���>�d�Z"�<5��^2���	x+9;�^g.XǠ>VIϋ�t��������;������0?��wy�~»��(*�<�}�^�>�>�;��5��>5�J��>�Ǐ+���>  @X̽�>���LY�?  �[�'�<�}�^�>���,.���Ǿ��{I'`���Ǐ+��þ  @X̽�����LY߿  �[�'���L�=q�>�;��5��>5�J��>,f�')��>  ����>�0bc @߿  ��m��<Usx�F6�>_�,9���>�0��&F�>X`���>�^g.XǠ>������0?��(*�<�ik��t�=�1�ډ|e>-�^��t�=�z=Ԉ|e>-�Uq�=���ye>���Tq�=��A�ye>�^LH�@�k
�8�࿶4_��&@����0,��