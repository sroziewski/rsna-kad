rsna_phase('dinov2', 'START')
main()
log('done')

_speed_dino_templates.clear()
gc.collect()

rsna_phase('dinov2', 'COMPLETE')