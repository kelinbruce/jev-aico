import unittest
from nimble.training.build_mixed_evidence import split_rows


class MixedSplitTests(unittest.TestCase):
    def test_family_partition_keeps_cross_release_variants_together(self):
        def row(id,family):
            return {'id':id,'family':id+'-pair','source_family':family,'split':'train',
                    'provenance':{'source_id':family},'input':{'state':id}}
        rows=[row('sol-outer','outer'),row('luna-outer','outer'),row('sonnet-inner','inner'),row('sol-train','train')]
        full,outer,train,val=split_rows(rows,{'outer'},{'inner'})
        self.assertEqual([r['id'] for r in outer],['sol-outer','luna-outer'])
        self.assertEqual([r['id'] for r in train],['sol-train'])
        self.assertEqual({r['id'] for r in full},{r['id'] for r in train+val})
        with self.assertRaisesRegex(ValueError,'overlap'):
            split_rows(rows,{'outer'},{'outer'})


if __name__=='__main__':
    unittest.main()
